"""Valuation dispatch and state-to-market risk conversion from the instructor template."""
from typing import Any
import numpy as np
from fixedincomelib.model import Model
from fixedincomelib.product import Product
from fixedincomelib.valuation.report import PVCashReport, RiskReprt, RiskReport
from fixedincomelib.valuation.valuation_engine import ValuationEngineProduct, ValuationRequest
from fixedincomelib.valuation.valuation_engine_registry import ValuationEngineProductRegistry
from fixedincomelib.valuation.valuation_parameters import ValuationParametersCollection

def create_value_report(
    model : Model,
    product : Product,
    valuation_parameters_collection: ValuationParametersCollection,
    request : ValuationRequest) -> Any:

    engine = ValuationEngineProductRegistry.new_valuation_engine(
        model,
        product,
        valuation_parameters_collection,
        request)

    engine.calculate_value()
    if request in [ValuationRequest.PV_DETAILED, ValuationRequest.PV, ValuationRequest.CASH]:
        v : PVCashReport = engine.get_value_and_cash()
        if request == ValuationRequest.PV_DETAILED:
            return v
        else:
            return v.pv if request == ValuationRequest.PV else v.cash
    elif request == ValuationRequest.FIRST_ORDER_RISK:
        return risk_calculation(engine)
    elif request == ValuationRequest.CASHFLOWS_REPORT:
        return engine.create_cash_flows_report()
    elif request == ValuationRequest.PAR_RATE_OR_SPREAD:
        return engine.par_rate_or_spread()
    else:
        raise Exception(f'Request is not currently supported.')


def risk_calculation(engine : ValuationEngineProduct):
    gradient = []
    engine.calculate_first_order_risk(gradient, 1., False)
    # Preserve the original assignment's direct-IFR report and state_gradient API.
    if not any(component.calibration_product for component in engine.model.components_.values()):
        return RiskReport.from_model_gradient(engine.model, gradient)
    # dV/dX^M = dV/dX^I \cdot J^{-1}, where J is the model jacobian dX^M/dX^I
    # step 1: flatten gradient
    gradient_flat = np.concatenate(gradient, axis=0) # dim : 1 x len(X^I)
    # step 2: get model (lazy) jacobian
    engine.model.calculate_model_jacobian()
    # TODO 4: Set risk to the row gradient with respect to X^M, using
    # gradient_flat and engine.model.model_jacobian as in the template.
    # Keep raw decimal-rate units; report display applies the bp scaling.
    # risk = gradient_flat @ J^{-1}; solving J^T risk^T = gradient^T avoids an explicit inverse.
    risk = np.linalg.solve(engine.model.model_jacobian.T, gradient_flat)
    # step 5: make a nice report
    risk_report = RiskReprt(engine.model.risk_postprocess(risk))
    return risk_report
