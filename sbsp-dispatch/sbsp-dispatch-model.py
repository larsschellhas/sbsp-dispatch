import pandas as pd
from datetime import datetime
from pyomo.environ import *
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Market:
    name = ""
    price_data = pd.Series()

    def __init__(self, name: str, price_data: pd.Series): 
         self.name = name
         self.price_data = price_data
        
def load_market_data(file_path: str) -> list[Market]:
    """Read and process market data from CSV and create Market instances.
    
    Args:
        file_path: Path to CSV file containing market data
        
    Returns:
        List of Market instances, one for each price column in the CSV
    """
    df = pd.read_csv(file_path, delimiter=',', parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    df.fillna(-10000, inplace=True)
    
    markets = []
    for column in df.columns:
        market = Market(name=column, price_data=df[column])
        markets.append(market)
        
    return markets, df.index

def create_model(time_steps: list[datetime], markets: list[Market], total_power_MW: int) -> ConcreteModel: 
    """Create a Pyomo model for the SBSP dispatch problem.
    
    Args:
        time_steps: List of datetime objects representing the time steps for the model
        markets: List of Market instances representing the markets to be considered
        total_power_MW: Total power capacity of the system, in MW
        
    Returns:
        Pyomo ConcreteModel instance representing the SBSP dispatch problem
    """

    logger.info(f"Creating optimization model with {len(time_steps)} time steps and {len(markets)} markets")
    start_time = time.time()

    model = ConcreteModel()

    # Sets
    model.T = Set(initialize=time_steps)
    model.MARKETS = Set(initialize=markets)

    model.INDEX_SET = Set(initialize=[(t, m) for t in model.T for m in model.MARKETS if m.price_data[t] > 0], dimen=2)

    # Precompute dispatch indices by time for faster constraint building
    dispatch_indices_by_time = {}
    for t in model.T:
        dispatch_indices_by_time[t] = [m for (time, m) in model.INDEX_SET if time == t]

    logger.info(f"Created {len(model.INDEX_SET.data())} index sets instead of {len(model.T) * len(model.MARKETS)} sets (Reduction by {(1 - len(model.INDEX_SET.data()) / len(model.T) / len(model.MARKETS)) * 100:.2f}%)")

    # Variables
    model.dispatch = Var(model.INDEX_SET, bounds=(0, None))
    model.revenue = Var(model.INDEX_SET, bounds=(None, None))

    # Constraints
    def power_limit_rule(model, t):
        return sum(model.dispatch[t, m] for m in dispatch_indices_by_time[t]) <= total_power_MW
    model.power_limit = Constraint(dispatch_indices_by_time.keys(), rule=power_limit_rule)

    def revenue_rule(model, t, m):  
        return model.revenue[t, m] == model.dispatch[t, m] * m.price_data[t]
    model.revenue_calculation = Constraint(model.INDEX_SET, rule=revenue_rule)

    # Objective
    def objective_rule(model):
        return sum(
            model.revenue[t, m]
            for (t, m) in model.INDEX_SET
        )
    model.objective = Objective(rule=objective_rule, sense=maximize)

    elapsed_time = time.time() - start_time
    logger.info(f"Model creation completed in {elapsed_time:.2f} seconds")
    return model


if __name__ == "__main__":
    total_start_time = time.time()
    
    # Load market data
    logger.info("Loading market data...")
    start_time = time.time()
    markets, time_steps = load_market_data("data/day_ahead_prices_all_domains_20150101-20241231.csv")
    elapsed_time = time.time() - start_time
    logger.info(f"Loaded {len(markets)} markets and {len(time_steps)} time steps in {elapsed_time:.2f} seconds")

    # Filter markets
    logger.info("Filtering markets and time steps...")
    start_time = time.time()
    # All markets
    # filtered_markets = [m for m in markets if m.name in ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE_AT_LU', 'DE_LU', 'DK_1', 'DK_2', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE_SEM', 'IT_BRNN', 'IT_CALA', 'IT_CNOR', 'IT_CSUD', 'IT_FOGN', 'IT_GR', 'IT_NORD', 'IT_NORD_AT', 'IT_NORD_CH', 'IT_NORD_FR', 'IT_NORD_SI', 'IT_PRGP', 'IT_ROSN', 'IT_SACO_AC', 'IT_SACO_DC', 'IT_SARD', 'IT_SICI', 'IT_SUD', 'LT', 'LV', 'ME', 'MK', 'NL', 'NO_1', 'NO_2', 'NO_2_NSL', 'NO_3', 'NO_4', 'NO_5', 'PL', 'PT', 'RO', 'RS', 'SE_1', 'SE_2', 'SE_3', 'SE_4', 'SI', 'SK', 'UA_BEI', 'UA_IPS']] 
    filtered_markets = [m for m in markets if m.name in ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE_AT_LU', 'DE_LU', 'DK_1', 'DK_2', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE_SEM', 'IT_BRNN', 'IT_CALA', 'IT_CNOR', 'IT_CSUD', 'IT_FOGN', 'IT_GR', 'IT_NORD', 'IT_NORD_AT', 'IT_NORD_CH', 'IT_NORD_FR', 'IT_NORD_SI', 'IT_PRGP', 'IT_ROSN', 'IT_SACO_AC', 'IT_SACO_DC', 'IT_SARD', 'IT_SICI', 'IT_SUD', 'LT', 'LV', 'ME', 'MK', 'NL', 'NO_1', 'NO_2', 'NO_2_NSL', 'NO_3', 'NO_4', 'NO_5', 'PL', 'PT', 'RO', 'RS', 'SE_1', 'SE_2', 'SE_3', 'SE_4', 'SI', 'SK']] 
    filtered_time_steps = [t for t in time_steps if (t.year == 2024 and t.month <= 6)]
    elapsed_time = time.time() - start_time
    logger.info(f"Selected {len(filtered_markets)} markets and {len(filtered_time_steps)} time steps in {elapsed_time:.2f} seconds")

    # Create model
    model = create_model(list(filtered_time_steps), filtered_markets, total_power_MW=1)

    # Solve
    logger.info("Starting optimization...")
    start_time = time.time()
    solver = SolverFactory('highs')
    results = solver.solve(model, tee=True)
    elapsed_time = time.time() - start_time
    logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")

    # Process results
    if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
        logger.info("Optimal solution found. Processing results...")
        start_time = time.time()
        
        # Extract results into a DataFrame
        dispatch_results = pd.DataFrame(index=filtered_time_steps)
        
        for market in filtered_markets:
            dispatch_results[f'{market.name}_dispatch'] = [model.dispatch[t, market].value 
                                                        for t in filtered_time_steps]
        
        # Save results to CSV
        out_file = f"data/dispatch_results_{filtered_time_steps[0].strftime('%Y%m%d')}-{filtered_time_steps[-1].strftime('%Y%m%d')}.csv"
        dispatch_results.to_csv(out_file)
        logger.info(f"Saved results to {out_file}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Results processing completed in {elapsed_time:.2f} seconds")

        total_elapsed_time = time.time() - total_start_time
        logger.info(f"Total execution time: {total_elapsed_time:.2f} seconds")
    else:
        logger.error(f"Solver status: {results.solver.status}")
        logger.error(f"Termination condition: {results.solver.termination_condition}")