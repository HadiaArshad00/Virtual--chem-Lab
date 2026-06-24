"""
Virtual Chemistry Lab API - Celery Tasks
Background task definitions for calculations and processing.
"""

import time
import traceback
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.workers.celery_app import celery
from app.db.session import AsyncSessionLocal
from app.models.experiment import Experiment
from app.models.result import Result
from app.models.batch import BatchJob
from app.core.calculators.dft import DFTCalculator
from app.core.calculators.kinetics import KineticsCalculator
from app.core.calculators.spectra import SpectraCalculator
from app.core.calculators.docking import DockingCalculator
from app.core.calculators.dynamics import DynamicsCalculator
from app.core.calculators.electrochem import ElectrochemCalculator
from app.core.calculators.crystallization import CrystallizationCalculator
from app.core.ml.yield_predictor import YieldPredictor
from app.core.ml.pka_predictor import PKaPredictor
from app.core.ml.logp_predictor import LogPPredictor
from app.core.ml.solvent_recommender import SolventRecommender
from app.core.engines.rdkit_engine import RDKitEngine
from app.core.utils.exceptions import ChemLabException


# Initialize calculators
dft_calc = DFTCalculator()
kinetics_calc = KineticsCalculator()
spectra_calc = SpectraCalculator()
docking_calc = DockingCalculator()
dynamics_calc = DynamicsCalculator()
electrochem_calc = ElectrochemCalculator()
crystallization_calc = CrystallizationCalculator()
yield_predictor = YieldPredictor()
pka_predictor = PKaPredictor()
logp_predictor = LogPPredictor()
solvent_recommender = SolventRecommender()
rdkit_engine = RDKitEngine()


import asyncio


def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop running
        return asyncio.run(coro)


@celery.task(bind=True, max_retries=3, default_retry_delay=60)
def calculate_experiment(self, experiment_id: int):
    """Calculate a single experiment.

    Args:
        experiment_id: ID of the experiment to calculate.

    Returns:
        Calculation result dictionary.
    """
    async def _run():
        async with AsyncSessionLocal() as db:
            try:
                # Get experiment
                result = await db.execute(
                    select(Experiment).where(Experiment.id == experiment_id)
                )
                experiment = result.scalar_one_or_none()

                if experiment is None:
                    return {"error": f"Experiment {experiment_id} not found"}

                # Update status to running
                experiment.status = "running"
                experiment.started_at = datetime.utcnow()
                await db.commit()

                # Run calculation based on type
                calc_result = await _run_calculation(experiment)

                # Update experiment with results
                experiment.status = "completed" if calc_result.success else "failed"
                experiment.results = calc_result.data
                experiment.completed_at = datetime.utcnow()
                experiment.engine_used = calc_result.engine_used
                experiment.calculation_time = calc_result.calculation_time

                if not calc_result.success:
                    experiment.error_message = calc_result.error_message

                await db.commit()

                # Save result record
                db_result = Result(
                    experiment_id=experiment_id,
                    output_data=calc_result.data,
                    engine_used=calc_result.engine_used,
                    calculation_time=calc_result.calculation_time,
                    warnings=calc_result.warnings,
                    citations=calc_result.citations,
                    error_message=calc_result.error_message,
                )
                db.add(db_result)
                await db.commit()

                return {
                    "experiment_id": experiment_id,
                    "status": experiment.status,
                    "success": calc_result.success,
                }

            except Exception as e:
                # Update experiment status to failed
                result = await db.execute(
                    select(Experiment).where(Experiment.id == experiment_id)
                )
                experiment = result.scalar_one_or_none()
                if experiment:
                    experiment.status = "failed"
                    experiment.error_message = str(e)
                    experiment.completed_at = datetime.utcnow()
                    await db.commit()

                # Retry on failure
                raise self.retry(exc=e)

    return run_async(_run())


async def _run_calculation(experiment: Experiment):
    """Run the appropriate calculation for an experiment.

    Args:
        experiment: Experiment model instance.

    Returns:
        CalculationResult.
    """
    params = experiment.parameters or {}
    exp_type = experiment.type

    if exp_type == "dft":
        calc_type = params.get("method", "geometry_optimization")
        if calc_type == "geometry_optimization":
            return await dft_calc.optimize_geometry(
                smiles=params.get("smiles", ""),
                functional=params.get("functional", "PBE0"),
                basis_set=params.get("basis_set", "def2-SVP"),
            )
        elif calc_type == "single_point":
            return await dft_calc.single_point_energy(
                smiles=params.get("smiles", ""),
                functional=params.get("functional", "B3LYP"),
                basis_set=params.get("basis_set", "6-31G*"),
            )
        elif calc_type == "ir_spectra":
            return await dft_calc.calculate_ir_spectra(
                smiles=params.get("smiles", ""),
            )
        elif calc_type == "nmr":
            return await dft_calc.calculate_nmr(
                smiles=params.get("smiles", ""),
                nuclei=params.get("nuclei", "1H"),
            )

    elif exp_type == "kinetics":
        calc_type = params.get("calculation_type", "arrhenius")
        if calc_type == "arrhenius":
            return await kinetics_calc.arrhenius_rate(
                A=params.get("A", 1e10),
                Ea=params.get("Ea", 50000),
                temperatures=params.get("temperatures", [298.15, 308.15, 318.15]),
            )
        elif calc_type == "eyring":
            return await kinetics_calc.eyring_rate(
                delta_G=params.get("delta_G", 80000),
                temperatures=params.get("temperatures", [298.15, 308.15, 318.15]),
            )

    elif exp_type == "spectra":
        spectra_type = params.get("spectra_type", "ir")
        if spectra_type == "ir":
            return await spectra_calc.calculate_ir(params.get("smiles", ""))
        elif spectra_type == "nmr":
            return await spectra_calc.calculate_nmr(
                params.get("smiles", ""),
                nuclei=params.get("nuclei", "1H"),
            )
        elif spectra_type == "mass_spec":
            return await spectra_calc.calculate_mass_spec(params.get("smiles", ""))
        elif spectra_type == "uv_vis":
            return await spectra_calc.calculate_uv_vis(params.get("smiles", ""))

    elif exp_type == "docking":
        return await docking_calc.dock(
            ligand_smiles=params.get("ligand_smiles", ""),
            receptor_pdbqt=params.get("receptor_pdbqt", ""),
        )

    elif exp_type == "dynamics":
        return await dynamics_calc.run_md(
            smiles=params.get("smiles", ""),
            num_steps=params.get("num_steps", 10000),
            temperature=params.get("temperature", 300.0),
        )

    elif exp_type == "electrochem":
        return await electrochem_calc.simulate_cv(
            E_start=params.get("E_start", -0.5),
            E_end=params.get("E_end", 0.5),
            scan_rate=params.get("scan_rate", 0.1),
        )

    elif exp_type == "crystallization":
        return await crystallization_calc.predict_crystallization(
            smiles=params.get("smiles", ""),
            temperature=params.get("temperature", 298.15),
            solvent=params.get("solvent", "water"),
        )

    elif exp_type == "yield_prediction":
        from app.core.engines.base import CalculationResult
        result = yield_predictor.predict(
            reactants_smiles=params.get("reactants_smiles", []),
            conditions=params.get("conditions", {}),
        )
        return CalculationResult(
            success=True,
            data=result,
            engine_used="ml",
            calculation_time=0.1,
            warnings=[],
            citations=[],
        )

    elif exp_type == "pka_prediction":
        from app.core.engines.base import CalculationResult
        result = pka_predictor.predict(params.get("smiles", ""))
        return CalculationResult(
            success=True,
            data=result,
            engine_used="ml",
            calculation_time=0.1,
            warnings=[],
            citations=[],
        )

    elif exp_type == "logp_prediction":
        from app.core.engines.base import CalculationResult
        result = logp_predictor.predict(params.get("smiles", ""))
        return CalculationResult(
            success=True,
            data=result,
            engine_used="ml",
            calculation_time=0.1,
            warnings=[],
            citations=[],
        )

    elif exp_type == "solvent_recommendation":
        from app.core.engines.base import CalculationResult
        result = solvent_recommender.recommend(
            reaction_type=params.get("reaction_type"),
            desired_properties=params.get("desired_properties", {}),
        )
        return CalculationResult(
            success=True,
            data=result,
            engine_used="ml",
            calculation_time=0.1,
            warnings=[],
            citations=[],
        )

    elif exp_type == "descriptors":
        return await rdkit_engine.calculate({
            "method": "descriptors",
            "smiles": params.get("smiles", ""),
        })

    elif exp_type == "fingerprint":
        return await rdkit_engine.calculate({
            "method": "morgan_fingerprint",
            "smiles": params.get("smiles", ""),
            "radius": params.get("radius", 2),
            "n_bits": params.get("n_bits", 2048),
        })

    elif exp_type == "similarity":
        return await rdkit_engine.calculate({
            "method": "similarity",
            "smiles": params.get("smiles", ""),
            "target_smiles": params.get("target_smiles", ""),
        })

    # Default: return error
    from app.core.engines.base import CalculationResult
    return CalculationResult(
        success=False,
        data={},
        engine_used="none",
        calculation_time=0.0,
        warnings=[],
        citations=[],
        error_message=f"Unknown experiment type: {exp_type}",
    )


@celery.task(bind=True, max_retries=2, default_retry_delay=120)
def process_batch(self, batch_id: int):
    """Process a batch job.

    Args:
        batch_id: ID of the batch job.

    Returns:
        Batch processing result.
    """
    async def _run():
        async with AsyncSessionLocal() as db:
            try:
                # Get batch job
                result = await db.execute(
                    select(BatchJob).where(BatchJob.id == batch_id)
                )
                batch = result.scalar_one_or_none()

                if batch is None:
                    return {"error": f"Batch job {batch_id} not found"}

                # Update status
                batch.status = "running"
                batch.started_at = datetime.utcnow()
                await db.commit()

                # Process each experiment
                results = []
                for exp_data in batch.experiments_data:
                    try:
                        # Create individual experiment
                        experiment = Experiment(
                            user_id=batch.user_id,
                            type=exp_data["type"],
                            status="pending",
                            parameters=exp_data["parameters"],
                        )
                        db.add(experiment)
                        await db.commit()
                        await db.refresh(experiment)

                        # Run calculation
                        calc_result = await _run_calculation(experiment)

                        # Update experiment
                        experiment.status = "completed" if calc_result.success else "failed"
                        experiment.results = calc_result.data
                        experiment.completed_at = datetime.utcnow()
                        experiment.engine_used = calc_result.engine_used
                        experiment.calculation_time = calc_result.calculation_time

                        if not calc_result.success:
                            experiment.error_message = calc_result.error_message

                        await db.commit()

                        # Update batch progress
                        if calc_result.success:
                            batch.completed_experiments += 1
                        else:
                            batch.failed_experiments += 1

                        results.append({
                            "experiment_id": experiment.id,
                            "type": experiment.type,
                            "status": experiment.status,
                            "success": calc_result.success,
                        })

                    except Exception as e:
                        batch.failed_experiments += 1
                        results.append({
                            "type": exp_data.get("type"),
                            "status": "failed",
                            "error": str(e),
                        })

                    await db.commit()

                # Update batch status
                batch.status = "completed"
                batch.completed_at = datetime.utcnow()
                batch.results = results
                await db.commit()

                return {
                    "batch_id": batch_id,
                    "status": "completed",
                    "total": batch.total_experiments,
                    "completed": batch.completed_experiments,
                    "failed": batch.failed_experiments,
                }

            except Exception as e:
                # Update batch status to failed
                result = await db.execute(
                    select(BatchJob).where(BatchJob.id == batch_id)
                )
                batch = result.scalar_one_or_none()
                if batch:
                    batch.status = "failed"
                    batch.error_message = str(e)
                    batch.completed_at = datetime.utcnow()
                    await db.commit()

                raise self.retry(exc=e)

    return run_async(_run())


@celery.task(bind=True, max_retries=3, default_retry_delay=30)
def export_results(self, experiment_id: int, format: str):
    """Export experiment results in specified format.

    Args:
        experiment_id: Experiment ID.
        format: Export format.

    Returns:
        Export result.
    """
    async def _run():
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Experiment).where(Experiment.id == experiment_id)
            )
            experiment = result.scalar_one_or_none()

            if experiment is None:
                return {"error": f"Experiment {experiment_id} not found"}

            if experiment.status != "completed":
                return {"error": f"Experiment {experiment_id} is not completed"}

            # Export logic would go here
            return {
                "experiment_id": experiment_id,
                "format": format,
                "status": "exported",
            }

    return run_async(_run())
