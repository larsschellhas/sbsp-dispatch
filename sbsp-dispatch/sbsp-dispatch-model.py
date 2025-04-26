import os
import pandas as pd
from datetime import datetime
from pyomo.environ import *
from collections import defaultdict
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
    """Create a Pyomo model for the SBSP dispatch problem."""

    logger.info(f"Creating optimization model with {len(time_steps)} time steps and {len(markets)} markets")
    total_start_time = time.time()

    # Initialize model
    start_time = time.time()
    model = ConcreteModel()
    logger.info(f"Model initialization: {time.time() - start_time:.2f} seconds")

    # Sets
    start_time = time.time()
    model.T = Set(initialize=time_steps)
    model.MARKETS = Set(initialize=markets)
    logger.info(f"Basic sets creation: {time.time() - start_time:.2f} seconds")

    # Index Set creation
    start_time = time.time()
    model.INDEX_SET = Set(initialize=[(t, m) for t in model.T for m in model.MARKETS if m.price_data[t] > 0], dimen=2)
    logger.info(f"INDEX_SET creation: {time.time() - start_time:.2f} seconds")

    # Precompute indices
    start_time = time.time()
    
    dispatch_indices_by_time = defaultdict(list)

    for (t, m) in model.INDEX_SET:
        dispatch_indices_by_time[t].append(m)
    logger.info(f"Dispatch indices precomputation: {time.time() - start_time:.2f} seconds")

    logger.info(f"Created {len(model.INDEX_SET.data())} index sets instead of {len(model.T) * len(model.MARKETS)} sets "
                f"(Reduction by {(1 - len(model.INDEX_SET.data()) / len(model.T) / len(model.MARKETS)) * 100:.2f}%)")

    # Variables
    start_time = time.time()
    model.dispatch = Var(model.INDEX_SET, bounds=(0, None))
    model.revenue = Var(model.INDEX_SET, bounds=(None, None))
    logger.info(f"Variables creation: {time.time() - start_time:.2f} seconds")

    # Constraints
    start_time = time.time()
    def power_limit_rule(model, t):
        return sum(model.dispatch[t, m] for m in dispatch_indices_by_time[t]) <= total_power_MW
    model.power_limit = Constraint(dispatch_indices_by_time.keys(), rule=power_limit_rule)
    logger.info(f"Power limit constraints creation: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    def revenue_rule(model, t, m):  
        return model.revenue[t, m] == model.dispatch[t, m] * m.price_data[t]
    model.revenue_calculation = Constraint(model.INDEX_SET, rule=revenue_rule)
    logger.info(f"Revenue constraints creation: {time.time() - start_time:.2f} seconds")

    # Objective
    start_time = time.time()
    def objective_rule(model):
        return sum(
            model.revenue[t, m]
            for (t, m) in model.INDEX_SET
        )
    model.objective = Objective(rule=objective_rule, sense=maximize)
    logger.info(f"Objective creation: {time.time() - start_time:.2f} seconds")

    total_elapsed_time = time.time() - total_start_time
    logger.info(f"Total model creation time: {total_elapsed_time:.2f} seconds")
    return model


def run_optimization(time_steps: list[datetime], markets: list[Market], total_power_MW: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run optimization for a specific time period and set of markets.
    
    Args:
        time_steps: List of datetime objects for the optimization period
        markets: List of Market objects to consider
        total_power_MW: Total power capacity in MW
        
    Returns:
        Tuple of (dispatch_results, revenue_results) as pandas DataFrames
    """
    start_time = time.time()
    logger.info(f"Starting optimization for {len(time_steps)} time steps and {len(markets)} markets")
    
    # Create and solve model
    model = create_model(time_steps, markets, total_power_MW)
    
    solver = SolverFactory('highs')
    results = solver.solve(model, tee=True)
    logger.info(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    
    if results.solver.status != 'ok' or results.solver.termination_condition != 'optimal':
        logger.error(f"Solver status: {results.solver.status}")
        logger.error(f"Termination condition: {results.solver.termination_condition}")
        raise RuntimeError("Optimization failed to find optimal solution")
    
    # Process results
    start_time = time.time()
    dispatch_results = pd.DataFrame(0.0, index=time_steps, 
                                  columns=[m.name for m in markets],
                                  dtype=float)
    revenue_results = pd.DataFrame(0.0, index=time_steps, 
                                 columns=[m.name for m in markets],
                                 dtype=float)
    
    for (t, m) in model.INDEX_SET:
        dispatch_results.at[t, m.name] = float(model.dispatch[t, m].value)
        revenue_results.at[t, m.name] = float(model.revenue[t, m].value)
    
    logger.info(f"Results processing completed in {time.time() - start_time:.2f} seconds")
    
    return dispatch_results, revenue_results

def save_results(dispatch_results: pd.DataFrame, revenue_results: pd.DataFrame, output_dir: str = "data"):
    """Save optimization results to CSV files."""
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames
    timestamp_start = dispatch_results.index[0].strftime('%Y%m%d')
    timestamp_end = dispatch_results.index[-1].strftime('%Y%m%d')
    
    dispatch_file = os.path.join(output_dir, f"dispatch_results_{timestamp_start}-{timestamp_end}.csv")
    revenue_file = os.path.join(output_dir, f"revenue_results_{timestamp_start}-{timestamp_end}.csv")
    
    # Save files
    dispatch_results.to_csv(dispatch_file)
    revenue_results.to_csv(revenue_file)
    
    logger.info(f"Saved results to {output_dir} in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    total_start_time = time.time()
    
    # Load market data
    logger.info("Loading market data...")
    start_time = time.time()
    markets, time_steps = load_market_data("data/day_ahead_prices_all_domains_20150101-20241231.csv")
    logger.info(f"Loaded {len(markets)} markets and {len(time_steps)} time steps in {time.time() - start_time:.2f} seconds")

    # Filter markets and time steps
    start_time = time.time()
    # All markets['AT', 'BE', 'BG', 'CH', 'CZ', 'DE_AT_LU', 'DE_LU', 'DK_1', 'DK_2', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE_SEM', 'IT_BRNN', 'IT_CALA', 'IT_CNOR', 'IT_CSUD', 'IT_FOGN', 'IT_GR', 'IT_NORD', 'IT_NORD_AT', 'IT_NORD_CH', 'IT_NORD_FR', 'IT_NORD_SI', 'IT_PRGP', 'IT_ROSN', 'IT_SACO_AC', 'IT_SACO_DC', 'IT_SARD', 'IT_SICI', 'IT_SUD', 'LT', 'LV', 'ME', 'MK', 'NL', 'NO_1', 'NO_2', 'NO_2_NSL', 'NO_3', 'NO_4', 'NO_5', 'PL', 'PT', 'RO', 'RS', 'SE_1', 'SE_2', 'SE_3', 'SE_4', 'SI', 'SK', 'UA_BEI', 'UA_IPS'] 
    filtered_markets = [m for m in markets if m.name in ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE_AT_LU', 'DE_LU', 'DK_1', 'DK_2', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE_SEM', 'IT_BRNN', 'IT_CALA', 'IT_CNOR', 'IT_CSUD', 'IT_FOGN', 'IT_GR', 'IT_NORD', 'IT_NORD_AT', 'IT_NORD_CH', 'IT_NORD_FR', 'IT_NORD_SI', 'IT_PRGP', 'IT_ROSN', 'IT_SACO_AC', 'IT_SACO_DC', 'IT_SARD', 'IT_SICI', 'IT_SUD', 'LT', 'LV', 'ME', 'MK', 'NL', 'NO_1', 'NO_2', 'NO_2_NSL', 'NO_3', 'NO_4', 'NO_5', 'PL', 'PT', 'RO', 'RS', 'SE_1', 'SE_2', 'SE_3', 'SE_4', 'SI', 'SK']] 
    filtered_time_steps = [t for t in time_steps if (t.year == 2024 and t.month > 11)]
    logger.info(f"Selected {len(filtered_markets)} markets and {len(filtered_time_steps)} time steps in {time.time() - start_time:.2f} seconds")

    # Run optimization
    try:
        dispatch_results, revenue_results = run_optimization(
            time_steps=filtered_time_steps,
            markets=filtered_markets,
            total_power_MW=1.0
        )
        
        # Save results
        save_results(dispatch_results, revenue_results)
        
        logger.info(f"Total execution time: {time.time() - total_start_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        sys.exit(1)