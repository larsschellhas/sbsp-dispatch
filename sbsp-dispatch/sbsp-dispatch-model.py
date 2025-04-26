import pandas as pd
import numpy as np
from datetime import datetime, date
from pyomo.environ import *


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

    # Create and solve optimization model
    model = ConcreteModel()

    # Sets
    model.T = Set(initialize=time_steps)
    model.MARKETS = Set(initialize=markets)

    # Variables
    model.dispatch = Var(model.MARKETS, model.T, bounds=(0, None))
    model.revenue = Var(model.MARKETS, model.T, bounds=(None, None))

    # Constraints
    def power_limit_rule(model, t):
        return sum(model.dispatch[m, t] for m in model.MARKETS) <= total_power_MW
    model.power_limit = Constraint(model.T, rule=power_limit_rule)

    def revenue_rule(model, market, t):
        return model.revenue[market, t] == model.dispatch[market, t] * market.price_data[t]

    # Objective
    def objective_rule(model):
        return sum(
            model.revenue[market, t]
            for market in model.MARKETS
            for t in model.T
        )
    model.objective = Objective(rule=objective_rule, sense=maximize)

    return model


if __name__ == "__main__":
    # Load market data
    markets, time_steps = load_market_data("data/day_ahead_prices_all_domains_20150101-20241231.csv")

    # All markets
    # filtered_markets = [m for m in markets if m.name in ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE_AT_LU', 'DE_LU', 'DK_1', 'DK_2', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE_SEM', 'IT_BRNN', 'IT_CALA', 'IT_CNOR', 'IT_CSUD', 'IT_FOGN', 'IT_GR', 'IT_NORD', 'IT_NORD_AT', 'IT_NORD_CH', 'IT_NORD_FR', 'IT_NORD_SI', 'IT_PRGP', 'IT_ROSN', 'IT_SACO_AC', 'IT_SACO_DC', 'IT_SARD', 'IT_SICI', 'IT_SUD', 'LT', 'LV', 'ME', 'MK', 'NL', 'NO_1', 'NO_2', 'NO_2_NSL', 'NO_3', 'NO_4', 'NO_5', 'PL', 'PT', 'RO', 'RS', 'SE_1', 'SE_2', 'SE_3', 'SE_4', 'SI', 'SK', 'UA_BEI', 'UA_IPS']] 

    # Filtered markets
    # filtered_markets = [m for m in markets if m.name in ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE_AT_LU', 'DE_LU', 'DK_1', 'DK_2', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE_SEM', 'IT_BRNN', 'IT_CALA', 'IT_CNOR', 'IT_CSUD', 'IT_FOGN', 'IT_GR', 'IT_NORD', 'IT_NORD_AT', 'IT_NORD_CH', 'IT_NORD_FR', 'IT_NORD_SI', 'IT_PRGP', 'IT_ROSN', 'IT_SACO_AC', 'IT_SACO_DC', 'IT_SARD', 'IT_SICI', 'IT_SUD', 'LT', 'LV', 'ME', 'MK', 'NL', 'NO_1', 'NO_2', 'NO_2_NSL', 'NO_3', 'NO_4', 'NO_5', 'PL', 'PT', 'RO', 'RS', 'SE_1', 'SE_2', 'SE_3', 'SE_4', 'SI', 'SK']] 
    filtered_markets = [m for m in markets if m.name in ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE_AT_LU', 'DE_LU']] 

    filtered_time_steps = [t for t in time_steps if (t.year == 2024 and t.month == 12)] 

    model = create_model(list(filtered_time_steps), filtered_markets, total_power_MW=1)
    pass
    # Solve
    solver = SolverFactory('highs')
    results =solver.solve(model)
    pass
    # Check if solution was optimal
    if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
        # Extract results into a DataFrame
        dispatch_results = pd.DataFrame(index=filtered_time_steps)
        
        for market in filtered_markets:
            dispatch_results[f'{market.name}_dispatch'] = [model.dispatch[market, t].value 
                                                        for t in filtered_time_steps]
            
        # Save results to CSV
        out_file = f"data/dispatch_results_{filtered_time_steps[0].strftime('%Y%m%d')}-{filtered_time_steps[-1].strftime('%Y%m%d')}.csv"
        dispatch_results.to_csv(out_file)
        print(f"Saved results to {out_file}")
        
        # Plot results
        import plotly.express as px
        fig = px.line(dispatch_results, x=filtered_time_steps, y=dispatch_results.columns, title='Dispatch Results')
        fig.show()