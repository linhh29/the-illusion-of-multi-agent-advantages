"""
Smfr price problem generation task.

This task generates problems based on historical smfr prices where entities
(investors) perform buy/sell transactions and we calculate totals or comparisons.
"""

import random
from datetime import datetime
from typing import Dict, Any, List, Optional
from task_base import BaseTask, BaseDataSource
from data_sources import SmfrDataSource


class SMFRTask(BaseTask):
    """
    Task for generating smfr price calculation problems.

    Generates haystack of historical smfr data and needles of investor
    transactions, then computes portfolio values.
    """

    TICKER_LIST = ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'ADBE', 'ABNB', 'AMZN', 'BIDU', 'COKE', 'DBX']
    COMPANY_LIST = ['Apple', 'Microsoft', 'Alphabet', 'Nvidia', 'Adobe', 'Airbnb',
                   'Amazon', 'Baidu', 'Coca-Cola', 'Dropbox']
    INVESTOR_LIST = ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward', 'Fiona', 'George',
                    'Helen', 'Isaac', 'Julia', 'Kevin', 'Laura', 'Michael', 'Nina',
                    'Oliver', 'Patricia', 'Quinn', 'Rachel', 'Samuel', 'Teresa']

    def __init__(self, config: Dict[str, Any], data_sources: Optional[List[BaseDataSource]] = None):
        """Initialize smfr task with config and data sources."""
        if data_sources is None:
            data_sources = [SmfrDataSource()]
        super().__init__(config, data_sources)

        # Extract task-specific parameters
        self.breadth = config.get('breadth', 3)
        self.depth = config.get('depth', 5)
        self.seed = config.get('seed', 100)

        task_params = config.get('task_params', {})
        self.price_type = task_params.get('price_type', 'Open')
        self.actions = task_params.get('actions', {
            'buy': ['bought', 'acquired', 'purchased'],
            'sell': ['sold', 'disposed']
        })

        # Question type: 'spending' (default), 'profit_loss', 'reverse_target_sell', 'reverse_target_buy'
        self.question_type = task_params.get('question_type', 'spending')

        # Target for reverse questions (percentage or amount)
        self.target_percentage = task_params.get('target_percentage', None)  # e.g., 10.0 for 10%
        self.target_amount = task_params.get('target_amount', None)  # e.g., 500.0 for $500

        # Number of distractor companies to include in haystack
        self.num_distractors = task_params.get('num_distractors', 0)

        # Override lists if provided in config
        self.ticker_list = task_params.get('ticker_list', self.TICKER_LIST)
        self.company_list = task_params.get('company_list', self.COMPANY_LIST)
        self.investor_list = task_params.get('investor_list', self.INVESTOR_LIST)

    def generate_haystack(self, seed: int) -> Dict[str, Any]:
        """
        Generate haystack of historical smfr price data.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing smfr data and metadata
        """
        random.seed(seed)

        # Select random smfr for the actual problem
        indices = random.sample(range(len(self.ticker_list)), k=self.breadth)
        selected_tickers = [self.ticker_list[i] for i in indices]
        selected_companies = [self.company_list[i] for i in indices]

        # Select distractor smfr (not used in the problem)
        distractor_indices = []
        distractor_tickers = []
        distractor_companies = []

        if self.num_distractors > 0:
            # Get remaining indices not used for actual problem
            available_indices = [i for i in range(len(self.ticker_list)) if i not in indices]

            # Sample distractor indices (up to available)
            num_to_select = min(self.num_distractors, len(available_indices))
            if num_to_select > 0:
                distractor_indices = random.sample(available_indices, k=num_to_select)
                distractor_tickers = [self.ticker_list[i] for i in distractor_indices]
                distractor_companies = [self.company_list[i] for i in distractor_indices]

        # Fetch smfr data for both selected and distractor smfr
        all_tickers = selected_tickers + distractor_tickers
        smfr_source = self.data_sources[0]
        smfr_data = smfr_source.fetch({
            'tickers': all_tickers,
            'days_back': 30
        })

        # Combine selected and distractor companies for interleaving
        all_companies_with_tickers = list(zip(
            selected_tickers + distractor_tickers,
            selected_companies + distractor_companies,
            [True] * len(selected_tickers) + [False] * len(distractor_tickers)  # True if used in problem
        ))

        # Shuffle to interleave distractors with actual companies
        random.shuffle(all_companies_with_tickers)

        # Format haystack text with interleaved companies
        haystack_text = ""
        for ticker, company, is_selected in all_companies_with_tickers:
            haystack_text += f"{company} Historical Smfr Price Data\n"

            records = smfr_data['data'][ticker]['records']
            for record in records:
                row_parts = []
                for key, value in record.items():
                    row_parts.append(f"{key}: {value}")
                haystack_text += " ".join(row_parts) + "\n"

            haystack_text += "\n"

        return {
            'text': haystack_text,
            'smfr_data': smfr_data,
            'selected_tickers': selected_tickers,
            'selected_companies': selected_companies,
            'distractor_tickers': distractor_tickers,
            'distractor_companies': distractor_companies,
            'indices': indices,
            'distractor_indices': distractor_indices,
            'seed': seed
        }

    def generate_needles(self, haystack: Dict[str, Any], seed: int, count: int = 1) -> List[Dict[str, Any]]:
        """
        Generate investor transaction needles.

        For portfolio-level reverse questions:
        - Generates completed buy-sell pairs across multiple smfr
        - Adds one incomplete transaction for a common smfr
        - reverse_target_sell: buy-only transaction (ask when to sell)
        - reverse_target_buy: sell-only transaction (ask when to buy)

        Args:
            haystack: Haystack data from generate_haystack()
            seed: Random seed for reproducibility
            count: Number of investor instances to generate

        Returns:
            List of needle dictionaries, one per investor
        """
        random.seed(seed)

        selected_tickers = haystack['selected_tickers']
        selected_companies = haystack['selected_companies']
        smfr_data = haystack['smfr_data']

        # Select investors
        investors = random.sample(self.investor_list, k=count)

        # Check if this is a portfolio-level reverse question
        is_portfolio_reverse = (
            self.question_type in ['reverse_target_sell', 'reverse_target_buy']
            and len(selected_tickers) > 1
        )

        # For portfolio-level reverse questions, select a common smfr
        common_smfr_idx = 0 if is_portfolio_reverse else None

        needles = []

        for investor in investors:
            investor_data = {
                'investor': investor,
                'transactions': [],
                'completed_transactions': [],  # Completed buy-sell pairs
                'incomplete_position': None,   # The common smfr position (buy or sell only)
                'portfolio_value': 0,
                'portfolio_cost': 0,           # Total cost of completed transactions
                'portfolio_revenue': 0,        # Total revenue from completed transactions
                'portfolio_profit': 0,         # Net profit from completed transactions
                'cot_steps': [],
                'haystack_ref': haystack
            }

            if is_portfolio_reverse:
                # Portfolio-level reverse question mode
                for smfr_idx, (ticker, company) in enumerate(zip(selected_tickers, selected_companies)):
                    records = smfr_data['data'][ticker]['records']

                    if smfr_idx == common_smfr_idx:
                        # This is the common smfr - generate incomplete transaction
                        # Smart date selection to reduce None occurrences
                        if self.question_type == 'reverse_target_sell':
                            # Buy early in the date range (more future dates available to sell)
                            # Use first 25% of date range
                            max_idx = max(1, int(len(records) * 0.25))
                            row_idx = random.choice(range(max_idx))
                            operation = 'buy'
                        else:  # reverse_target_buy
                            # Sell late in the date range (more past dates available to buy)
                            # Use last 25% of date range
                            min_idx = int(len(records) * 0.75)
                            row_idx = random.choice(range(min_idx, len(records)))
                            operation = 'sell'

                        record = records[row_idx]

                        date_str = record['Date'].split('T')[0].split('-')
                        formatted_date = datetime(
                            int(date_str[0]),
                            int(date_str[1]),
                            int(date_str[2])
                        ).strftime("%B %d, %Y")

                        shares = random.randint(50, 100)
                        price = record[self.price_type]
                        value = price * shares

                        incomplete_transaction = {
                            'investor': investor,
                            'action': random.choice(self.actions[operation]),
                            'operation': operation,
                            'number': shares,
                            'company': company,
                            'ticker': ticker,
                            'date': formatted_date,
                            'price': price,
                            'value': value,
                            'price_type': self.price_type,
                            'row_idx': row_idx,
                            'raw_date': record['Date']
                        }

                        investor_data['incomplete_position'] = incomplete_transaction
                        investor_data['transactions'].append(incomplete_transaction)

                    else:
                        # For other smfr, generate completed buy-sell pairs
                        num_pairs = self.depth // 2
                        rows = random.sample(range(len(records)), k=num_pairs * 2)
                        rows.sort()

                        for pair_idx in range(num_pairs):
                            buy_idx = pair_idx * 2
                            sell_idx = buy_idx + 1
                            shares = random.randint(50, 100)

                            # Buy transaction
                            buy_record = records[rows[buy_idx]]
                            buy_date_str = buy_record['Date'].split('T')[0].split('-')
                            buy_formatted_date = datetime(
                                int(buy_date_str[0]),
                                int(buy_date_str[1]),
                                int(buy_date_str[2])
                            ).strftime("%B %d, %Y")
                            buy_price = buy_record[self.price_type]
                            buy_value = buy_price * shares

                            buy_transaction = {
                                'investor': investor,
                                'action': random.choice(self.actions['buy']),
                                'operation': 'buy',
                                'number': shares,
                                'company': company,
                                'ticker': ticker,
                                'date': buy_formatted_date,
                                'price': buy_price,
                                'value': buy_value,
                                'price_type': self.price_type,
                                'row_idx': rows[buy_idx],
                                'raw_date': buy_record['Date']
                            }

                            # Sell transaction
                            sell_record = records[rows[sell_idx]]
                            sell_date_str = sell_record['Date'].split('T')[0].split('-')
                            sell_formatted_date = datetime(
                                int(sell_date_str[0]),
                                int(sell_date_str[1]),
                                int(sell_date_str[2])
                            ).strftime("%B %d, %Y")
                            sell_price = sell_record[self.price_type]
                            sell_value = sell_price * shares

                            sell_transaction = {
                                'investor': investor,
                                'action': random.choice(self.actions['sell']),
                                'operation': 'sell',
                                'number': shares,
                                'company': company,
                                'ticker': ticker,
                                'date': sell_formatted_date,
                                'price': sell_price,
                                'value': sell_value,
                                'price_type': self.price_type,
                                'row_idx': rows[sell_idx],
                                'raw_date': sell_record['Date']
                            }

                            # Calculate profit/loss for this pair
                            profit = sell_value - buy_value

                            # Update portfolio tracking
                            investor_data['portfolio_cost'] += buy_value
                            investor_data['portfolio_revenue'] += sell_value
                            investor_data['portfolio_profit'] += profit

                            # Store completed pair
                            completed_pair = {
                                'buy': buy_transaction,
                                'sell': sell_transaction,
                                'profit': profit,
                                'profit_percentage': (profit / buy_value * 100) if buy_value > 0 else 0
                            }
                            investor_data['completed_transactions'].append(completed_pair)

                            # Add to transactions list
                            investor_data['transactions'].append(buy_transaction)
                            investor_data['transactions'].append(sell_transaction)

            else:
                # Original behavior for spending, profit_loss, and simple reverse questions
                for ticker, company in zip(selected_tickers, selected_companies):
                    records = smfr_data['data'][ticker]['records']
                    rows = random.sample(range(len(records)), k=self.depth)
                    rows.sort()

                    for i, row_idx in enumerate(rows):
                        operation = 'buy' if i % 2 == 0 else 'sell'
                        action = random.choice(self.actions[operation])

                        if self.question_type in ['profit_loss', 'reverse_target_sell', 'reverse_target_buy']:
                            if i % 2 == 0:
                                current_shares = random.randint(50, 100)
                            else:
                                current_shares = investor_data['transactions'][-1]['number']
                            number = current_shares
                        else:
                            number = random.randint(50, 100) if i == 0 else random.randint(0, 100)

                        record = records[row_idx]
                        date_str = record['Date'].split('T')[0].split('-')
                        formatted_date = datetime(
                            int(date_str[0]),
                            int(date_str[1]),
                            int(date_str[2])
                        ).strftime("%B %d, %Y")

                        price = record[self.price_type]
                        value = price * number

                        transaction = {
                            'investor': investor,
                            'action': action,
                            'operation': operation,
                            'number': number,
                            'company': company,
                            'ticker': ticker,
                            'date': formatted_date,
                            'price': price,
                            'value': value,
                            'price_type': self.price_type,
                            'row_idx': row_idx,
                            'raw_date': record['Date']
                        }

                        investor_data['transactions'].append(transaction)

                        if operation == 'buy':
                            investor_data['portfolio_value'] += value

                            price_type_str = 'opening' if self.price_type == 'Open' else 'closing'
                            cot_step = (f"{company} {price_type_str} price on {formatted_date} = {price}. "
                                      f"{price} x {number} = {value}")
                            investor_data['cot_steps'].append(cot_step)

            needles.append(investor_data)

        return needles

    def _create_transaction_pairs(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pair up buy-sell transactions and calculate profit/loss.

        Args:
            transactions: List of transactions from a needle

        Returns:
            List of transaction pairs with profit/loss information
        """
        pairs = []

        # Group transactions by smfr ticker
        from collections import defaultdict
        by_ticker = defaultdict(list)
        for txn in transactions:
            by_ticker[txn['ticker']].append(txn)

        # For each smfr, pair buy-sell transactions
        for ticker, txn_list in by_ticker.items():
            # Transactions should alternate buy-sell
            for i in range(0, len(txn_list) - 1, 2):
                buy_txn = txn_list[i]
                sell_txn = txn_list[i + 1]

                # Verify this is actually a buy-sell pair
                if buy_txn['operation'] != 'buy' or sell_txn['operation'] != 'sell':
                    continue

                # Calculate profit/loss
                buy_total = buy_txn['price'] * buy_txn['number']
                sell_total = sell_txn['price'] * sell_txn['number']
                profit_amount = sell_total - buy_total
                profit_percentage = (profit_amount / buy_total) * 100 if buy_total > 0 else 0
                profit_per_share = sell_txn['price'] - buy_txn['price']

                pair = {
                    'ticker': ticker,
                    'company': buy_txn['company'],
                    'buy': {
                        'date': buy_txn['date'],
                        'raw_date': buy_txn['raw_date'],
                        'price': buy_txn['price'],
                        'shares': buy_txn['number'],
                        'total': buy_total,
                        'row_idx': buy_txn['row_idx']
                    },
                    'sell': {
                        'date': sell_txn['date'],
                        'raw_date': sell_txn['raw_date'],
                        'price': sell_txn['price'],
                        'shares': sell_txn['number'],
                        'total': sell_total,
                        'row_idx': sell_txn['row_idx']
                    },
                    'profit_loss': {
                        'amount': profit_amount,
                        'percentage': profit_percentage,
                        'per_share': profit_per_share
                    }
                }

                pairs.append(pair)

        return pairs

    def _find_valid_dates_for_target(self, haystack: Dict[str, Any], pair: Dict[str, Any],
                                     target_type: str) -> List[str]:
        """
        Find all dates where the target profit/loss could be achieved.

        Args:
            haystack: Haystack data with smfr records
            pair: A transaction pair with buy/sell info
            target_type: 'sell' (find sell dates given buy) or 'buy' (find buy dates given sell)

        Returns:
            List of valid dates as formatted strings
        """
        ticker = pair['ticker']
        smfr_data = haystack['smfr_data']
        records = smfr_data['data'][ticker]['records']

        valid_dates = []

        if target_type == 'sell':
            # Given a buy, find all dates to sell for target profit
            buy_price = pair['buy']['price']
            shares = pair['buy']['shares']
            buy_date_raw = pair['buy']['raw_date']

            # Calculate target sell price
            if self.target_percentage is not None:
                # Percentage-based target
                target_sell_price = buy_price * (1 + self.target_percentage / 100)
            elif self.target_amount is not None:
                # Amount-based target
                target_sell_price = buy_price + (self.target_amount / shares)
            else:
                return []

            # Find all dates after buy date with price >= target
            for record in records:
                record_date_raw = record['Date']
                # Only consider dates after the buy date
                if record_date_raw > buy_date_raw:
                    sell_price = record[self.price_type]
                    if sell_price >= target_sell_price:
                        # Format the date
                        date_str = record['Date'].split('T')[0].split('-')
                        formatted_date = datetime(
                            int(date_str[0]),
                            int(date_str[1]),
                            int(date_str[2])
                        ).strftime("%B %d, %Y")
                        valid_dates.append(formatted_date)

        elif target_type == 'buy':
            # Given a sell, find all dates to buy for target profit
            sell_price = pair['sell']['price']
            shares = pair['sell']['shares']
            sell_date_raw = pair['sell']['raw_date']

            # Calculate target buy price
            if self.target_percentage is not None:
                # For percentage: sell_price = buy_price * (1 + pct/100)
                # So: buy_price = sell_price / (1 + pct/100)
                target_buy_price = sell_price / (1 + self.target_percentage / 100)
            elif self.target_amount is not None:
                # Amount-based target
                target_buy_price = sell_price - (self.target_amount / shares)
            else:
                return []

            # Find all dates before sell date with price <= target
            for record in records:
                record_date_raw = record['Date']
                # Only consider dates before the sell date
                if record_date_raw < sell_date_raw:
                    buy_price = record[self.price_type]
                    if buy_price <= target_buy_price:
                        # Format the date
                        date_str = record['Date'].split('T')[0].split('-')
                        formatted_date = datetime(
                            int(date_str[0]),
                            int(date_str[1]),
                            int(date_str[2])
                        ).strftime("%B %d, %Y")
                        valid_dates.append(formatted_date)

        return valid_dates

    def _find_valid_dates_portfolio_level(self, haystack: Dict[str, Any], needle: Dict[str, Any],
                                          target_type: str) -> List[str]:
        """
        Find all dates where the portfolio target could be achieved.

        For portfolio-level questions:
        - Calculate existing portfolio profit from completed transactions
        - Determine what profit is needed from incomplete position
        - Find dates where that profit can be achieved

        Args:
            haystack: Haystack data with smfr records
            needle: Investor's complete portfolio data
            target_type: 'sell' (find sell dates for held smfr) or 'buy' (find buy dates)

        Returns:
            List of valid dates as formatted strings
        """
        incomplete = needle['incomplete_position']
        if not incomplete:
            return []

        ticker = incomplete['ticker']
        smfr_data = haystack['smfr_data']
        records = smfr_data['data'][ticker]['records']
        shares = incomplete['number']

        # Calculate total portfolio cost
        # For reverse_target_sell: cost = completed cost + incomplete buy cost
        # For reverse_target_buy: cost = completed cost - incomplete sell revenue (we didn't buy yet)
        if target_type == 'sell':
            # We already bought the common smfr, now finding when to sell
            incomplete_cost = incomplete['value']
            total_cost = needle['portfolio_cost'] + incomplete_cost
        else:  # target_type == 'buy'
            # We already sold the common smfr, now finding when we could have bought
            incomplete_revenue = incomplete['value']
            # Cost is what we paid for everything else; we'll add buy cost
            total_cost = needle['portfolio_cost']

        # Calculate target profit amount
        if self.target_percentage is not None:
            target_profit = total_cost * (self.target_percentage / 100)
        elif self.target_amount is not None:
            target_profit = self.target_amount
        else:
            return []

        # Profit needed from incomplete position
        # existing_profit + incomplete_profit = target_profit
        existing_profit = needle['portfolio_profit']
        needed_profit = target_profit - existing_profit

        valid_dates = []

        if target_type == 'sell':
            # Given a buy, find all dates to sell for needed profit
            buy_price = incomplete['price']
            buy_date_raw = incomplete['raw_date']

            # Calculate required sell price
            # needed_profit = (sell_price * shares) - (buy_price * shares)
            # needed_profit = shares * (sell_price - buy_price)
            # sell_price = buy_price + (needed_profit / shares)
            required_sell_price = buy_price + (needed_profit / shares)

            # Find all dates after buy date with price >= required
            for record in records:
                record_date_raw = record['Date']
                if record_date_raw > buy_date_raw:
                    sell_price = record[self.price_type]
                    if sell_price >= required_sell_price:
                        date_str = record['Date'].split('T')[0].split('-')
                        formatted_date = datetime(
                            int(date_str[0]),
                            int(date_str[1]),
                            int(date_str[2])
                        ).strftime("%B %d, %Y")
                        valid_dates.append(formatted_date)

        elif target_type == 'buy':
            # Given a sell, find all dates to buy for needed profit
            sell_price = incomplete['price']
            sell_date_raw = incomplete['raw_date']
            sell_revenue = incomplete['value']

            # For reverse_target_buy, we need to account for the buy cost in total portfolio cost
            # total_cost_with_buy = total_cost + (buy_price * shares)
            # target_profit = total_cost_with_buy * (target_percentage / 100)
            # total_revenue = portfolio_revenue + sell_revenue
            # profit = total_revenue - total_cost_with_buy
            # We want: profit = target_profit
            # total_revenue - total_cost_with_buy = target_profit
            # (portfolio_revenue + sell_revenue) - (portfolio_cost + buy_price*shares) = target_profit
            # sell_revenue - buy_price*shares = target_profit - (portfolio_revenue - portfolio_cost)
            # sell_revenue - buy_price*shares = target_profit - existing_profit
            # -buy_price*shares = target_profit - existing_profit - sell_revenue
            # buy_price*shares = sell_revenue - target_profit + existing_profit

            # Wait, this is circular because target depends on total cost which depends on buy price
            # Let's use target_amount instead for buy questions, or recalculate properly

            # Actually for percentage targets in reverse_target_buy:
            # Let B = buy_price * shares (what we pay)
            # Total cost = portfolio_cost + B
            # Total revenue = portfolio_revenue + sell_revenue
            # Total profit = Total revenue - Total cost = (portfolio_revenue + sell_revenue) - (portfolio_cost + B)
            # Total profit = existing_profit + sell_revenue - B
            # We want: Total profit / Total cost = target_percentage / 100
            # (existing_profit + sell_revenue - B) / (portfolio_cost + B) = target_percentage / 100
            # existing_profit + sell_revenue - B = (target_percentage / 100) * (portfolio_cost + B)
            # existing_profit + sell_revenue - B = (target_percentage * portfolio_cost / 100) + (target_percentage * B / 100)
            # existing_profit + sell_revenue = B - (target_percentage * B / 100) + (target_percentage * portfolio_cost / 100)
            # existing_profit + sell_revenue = B * (1 - target_percentage/100) + (target_percentage * portfolio_cost / 100)
            # B * (1 - target_percentage/100) = existing_profit + sell_revenue - (target_percentage * portfolio_cost / 100)
            # B = (existing_profit + sell_revenue - (target_percentage * portfolio_cost / 100)) / (1 - target_percentage/100)

            if self.target_percentage is not None:
                target_buy_cost = (
                    (existing_profit + sell_revenue - (self.target_percentage * total_cost / 100))
                    / (1 - self.target_percentage / 100)
                )
                required_buy_price = target_buy_cost / shares
            elif self.target_amount is not None:
                # For fixed amount target:
                # Total profit = target_amount
                # existing_profit + sell_revenue - B = target_amount
                # B = existing_profit + sell_revenue - target_amount
                target_buy_cost = existing_profit + sell_revenue - self.target_amount
                required_buy_price = target_buy_cost / shares
            else:
                return []

            # Find all dates before sell date with price <= required
            for record in records:
                record_date_raw = record['Date']
                if record_date_raw < sell_date_raw:
                    buy_price = record[self.price_type]
                    if buy_price <= required_buy_price:
                        date_str = record['Date'].split('T')[0].split('-')
                        formatted_date = datetime(
                            int(date_str[0]),
                            int(date_str[1]),
                            int(date_str[2])
                        ).strftime("%B %d, %Y")
                        valid_dates.append(formatted_date)

        return valid_dates

    def compute_answer(self, needle: Dict[str, Any]) -> Any:
        """
        Compute the answer for a single needle based on question type.

        Args:
            needle: Single investor's transaction data

        Returns:
            Answer depends on question_type:
            - 'spending': Total amount spent (float)
            - 'profit_loss': Total profit/loss (float)
            - 'reverse_target_sell': List of valid sell dates (portfolio or single-smfr)
            - 'reverse_target_buy': List of valid buy dates (portfolio or single-smfr)
        """
        if self.question_type == 'spending':
            # Original behavior: return total spending
            return needle['portfolio_value']

        elif self.question_type == 'profit_loss':
            # Calculate total profit/loss from all transaction pairs
            pairs = self._create_transaction_pairs(needle['transactions'])
            total_profit_loss = sum(pair['profit_loss']['amount'] for pair in pairs)
            # Store pairs in needle for later use in formatting
            needle['transaction_pairs'] = pairs
            return total_profit_loss

        elif self.question_type == 'reverse_target_sell':
            # Check if this is portfolio-level or single-smfr
            if needle.get('incomplete_position') is not None:
                # Portfolio-level: find dates to sell common smfr to achieve portfolio target
                valid_dates = self._find_valid_dates_portfolio_level(
                    needle.get('haystack_ref'),
                    needle,
                    'sell'
                )
            else:
                # Single-smfr (legacy): find dates across all pairs
                pairs = self._create_transaction_pairs(needle['transactions'])
                needle['transaction_pairs'] = pairs

                all_valid_dates = []
                for pair in pairs:
                    valid_dates = self._find_valid_dates_for_target(
                        needle.get('haystack_ref'),
                        pair,
                        'sell'
                    )
                    all_valid_dates.extend(valid_dates)

                valid_dates = all_valid_dates

            # Remove duplicates and sort
            unique_dates = sorted(list(set(valid_dates)),
                                key=lambda d: datetime.strptime(d, "%B %d, %Y"))
            return unique_dates

        elif self.question_type == 'reverse_target_buy':
            # Check if this is portfolio-level or single-smfr
            if needle.get('incomplete_position') is not None:
                # Portfolio-level: find dates to buy common smfr to achieve portfolio target
                valid_dates = self._find_valid_dates_portfolio_level(
                    needle.get('haystack_ref'),
                    needle,
                    'buy'
                )
            else:
                # Single-smfr (legacy): find dates across all pairs
                pairs = self._create_transaction_pairs(needle['transactions'])
                needle['transaction_pairs'] = pairs

                all_valid_dates = []
                for pair in pairs:
                    valid_dates = self._find_valid_dates_for_target(
                        needle.get('haystack_ref'),
                        pair,
                        'buy'
                    )
                    all_valid_dates.extend(valid_dates)

                valid_dates = all_valid_dates

            # Remove duplicates and sort
            unique_dates = sorted(list(set(valid_dates)),
                                key=lambda d: datetime.strptime(d, "%B %d, %Y"))
            return unique_dates

        else:
            # Default to spending
            return needle['portfolio_value']

    def format_problem(self, haystack: Dict[str, Any], needles: List[Dict[str, Any]],
                      question_template: str, extra_vars: Optional[Dict[str, str]] = None) -> str:
        """
        Format the complete problem text.

        Args:
            haystack: Haystack data
            needles: List of investor transaction data
            question_template: Template string with placeholders
            extra_vars: Optional extra template variables (e.g., comparison_word)

        Returns:
            Formatted problem string
        """
        # Collect all transactions from all investors
        all_transactions = []
        for needle in needles:
            for txn in needle['transactions']:
                all_transactions.append(txn)

        # Sort transactions by date
        # Parse date strings for sorting (format: "Month DD, YYYY")
        def parse_date_for_sort(txn):
            date_str = txn['date']
            # Convert "January 01, 2024" to datetime for sorting
            from datetime import datetime
            return datetime.strptime(date_str, "%B %d, %Y")

        all_transactions.sort(key=parse_date_for_sort)

        # Group transactions by date
        from collections import defaultdict
        transactions_by_date = defaultdict(list)
        for txn in all_transactions:
            transactions_by_date[txn['date']].append(txn)

        # Format transactions grouped by date with date headers
        transaction_paragraphs = []
        for date in sorted(transactions_by_date.keys(), key=lambda d: parse_date_for_sort({'date': d})):
            # Add date header
            paragraph = f"{date}:\n"

            # Group transactions by investor
            from collections import defaultdict
            investor_txns = defaultdict(list)
            for txn in transactions_by_date[date]:
                investor_txns[txn['investor']].append(txn)

            # Format consolidated transactions - one sentence per investor
            txn_sentences = []
            for investor in sorted(investor_txns.keys()):
                # Group this investor's transactions by action
                action_groups = defaultdict(list)
                for txn in investor_txns[investor]:
                    action_groups[txn['action']].append(txn)

                # Build action clauses for this investor
                action_clauses = []
                for action in sorted(action_groups.keys()):
                    txns = action_groups[action]
                    # Build smfr list for this action
                    smfr_parts = [f"{txn['number']} shares of {txn['company']}" for txn in txns]

                    # Join smfr with commas and 'and'
                    if len(smfr_parts) == 1:
                        smfr_str = smfr_parts[0]
                    elif len(smfr_parts) == 2:
                        smfr_str = f"{smfr_parts[0]} and {smfr_parts[1]}"
                    else:
                        smfr_str = ', '.join(smfr_parts[:-1]) + f", and {smfr_parts[-1]}"

                    action_clauses.append(f"{action} {smfr_str}")

                # Join all actions for this investor
                if len(action_clauses) == 1:
                    investor_sentence = f"{investor} {action_clauses[0]}"
                elif len(action_clauses) == 2:
                    investor_sentence = f"{investor} {action_clauses[0]} and {action_clauses[1]}"
                else:
                    investor_sentence = f"{investor} {', '.join(action_clauses[:-1])}, and {action_clauses[-1]}"

                txn_sentences.append(investor_sentence)

            # Join sentences with periods
            paragraph += '. '.join(txn_sentences) + '.'
            transaction_paragraphs.append(paragraph)

        # Join paragraphs with double newlines
        transaction_text = '\n\n'.join(transaction_paragraphs)

        # Build template variables
        template_vars = {
            'haystack': haystack['text'],
            'price_type': 'opening' if self.price_type == 'Open' else 'closing',
            'entity': needles[0]['investor'] if len(needles) > 0 else '',
            'actions': transaction_text,
            'all_actions': transaction_text
        }

        # For multi-instance problems, also provide indexed versions
        # (though these will be the same as 'actions' since we're sorting by date)
        for i in range(len(needles)):
            template_vars[f'actions_{i}'] = transaction_text

        # Merge in extra variables if provided
        if extra_vars:
            template_vars.update(extra_vars)

        return question_template.format(**template_vars)

    def format_cot(self, needles: List[Dict[str, Any]], answers: List[Any]) -> str:
        """
        Format chain-of-thought reasoning.

        For portfolio-level reverse questions, shows:
        - Completed transaction pairs with their P/L
        - Portfolio-level profit calculation
        - Incomplete position details
        - Required price calculation
        - Valid dates

        Args:
            needles: List of investor transaction data
            answers: List of computed answers

        Returns:
            Formatted chain-of-thought string
        """
        cot_parts = []

        for needle, answer in zip(needles, answers):
            cot = f"{needle['investor']}\n"

            # Check if this is portfolio-level reverse question
            is_portfolio_reverse = (
                self.question_type in ['reverse_target_sell', 'reverse_target_buy']
                and needle.get('incomplete_position') is not None
            )

            if is_portfolio_reverse:
                # Portfolio-level CoT
                price_type_str = 'opening' if self.price_type == 'Open' else 'closing'

                # Show completed transactions
                cot += "Completed transactions:\n"
                for pair in needle['completed_transactions']:
                    buy_txn = pair['buy']
                    sell_txn = pair['sell']
                    profit = pair['profit']

                    cot += f"  {buy_txn['company']}: Bought {buy_txn['number']} shares at ${buy_txn['price']:.2f} on {buy_txn['date']}, "
                    cot += f"sold at ${sell_txn['price']:.2f} on {sell_txn['date']}. "
                    cot += f"Profit: ${profit:.2f}\n"

                # Show portfolio summary
                cot += f"\nPortfolio cost: ${needle['portfolio_cost']:.2f}\n"
                cot += f"Portfolio revenue: ${needle['portfolio_revenue']:.2f}\n"
                cot += f"Portfolio profit from completed transactions: ${needle['portfolio_profit']:.2f}\n"

                # Show incomplete position
                incomplete = needle['incomplete_position']
                cot += f"\n{incomplete['company']} position: "
                if self.question_type == 'reverse_target_sell':
                    cot += f"Bought {incomplete['number']} shares at ${incomplete['price']:.2f} on {incomplete['date']}\n"
                    total_cost = needle['portfolio_cost'] + incomplete['value']
                    cot += f"Total portfolio cost: ${total_cost:.2f}\n"
                else:  # reverse_target_buy
                    cot += f"Sold {incomplete['number']} shares at ${incomplete['price']:.2f} on {incomplete['date']}\n"
                    total_cost = needle['portfolio_cost']

                # Show target calculation
                if self.target_percentage is not None:
                    target_profit = total_cost * (self.target_percentage / 100)
                    cot += f"Target: {self.target_percentage}% profit = ${target_profit:.2f}\n"
                elif self.target_amount is not None:
                    target_profit = self.target_amount
                    cot += f"Target profit: ${target_profit:.2f}\n"

                needed_profit = target_profit - needle['portfolio_profit']
                cot += f"Profit needed from {incomplete['company']}: ${needed_profit:.2f}\n"

                # Show required price
                if self.question_type == 'reverse_target_sell':
                    required_price = incomplete['price'] + (needed_profit / incomplete['number'])
                    cot += f"Required sell price: ${required_price:.2f} or higher\n"
                else:  # reverse_target_buy
                    # This is more complex for percentage targets
                    if self.target_percentage is not None:
                        sell_revenue = incomplete['value']
                        target_buy_cost = (
                            (needle['portfolio_profit'] + sell_revenue - (self.target_percentage * total_cost / 100))
                            / (1 - self.target_percentage / 100)
                        )
                        required_price = target_buy_cost / incomplete['number']
                    else:
                        target_buy_cost = needle['portfolio_profit'] + incomplete['value'] - target_profit
                        required_price = target_buy_cost / incomplete['number']
                    cot += f"Required buy price: ${required_price:.2f} or lower\n"

                cot += f"\nValid dates: {answer}\n"

            else:
                # Original CoT for non-portfolio questions
                cot += '\n'.join(needle['cot_steps'])

                # Format the answer based on question type
                if self.question_type == 'spending':
                    cot += f"\nTotal spent: {answer}\n"
                elif self.question_type == 'profit_loss':
                    cot += f"\nTotal profit/loss: {answer}\n"
                elif self.question_type in ['reverse_target_sell', 'reverse_target_buy']:
                    # For reverse questions, the answer is a list of dates
                    cot += f"\nValid dates: {answer}\n"
                else:
                    # Default fallback
                    cot += f"\nAnswer: {answer}\n"

            cot_parts.append(cot)

        return '\n\n'.join(cot_parts)

    def get_task_type(self) -> str:
        """Return task type identifier."""
        return "smfr"
