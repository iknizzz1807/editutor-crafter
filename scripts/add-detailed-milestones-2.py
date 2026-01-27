#!/usr/bin/env python3
"""Add detailed milestones for remaining new projects."""

import yaml
from pathlib import Path

DETAILED_PROJECTS = {
    "build-graphql-engine": {
        "name": "Build Your Own GraphQL Engine",
        "description": "Build a GraphQL execution engine with query parsing, schema stitching, and automatic API generation like Hasura/PostGraphile",
        "category": "Backend Infrastructure",
        "difficulty": "expert",
        "estimated_hours": 120,
        "skills": [
            "GraphQL spec",
            "Query parsing",
            "Schema introspection",
            "Database reflection",
            "Query optimization",
            "Real-time subscriptions"
        ],
        "prerequisites": ["GraphQL server", "Database internals", "Compiler basics"],
        "learning_outcomes": [
            "Deep understanding of GraphQL specification",
            "Implement query parsing and validation",
            "Build automatic schema generation from database",
            "Optimize GraphQL queries to SQL"
        ],
        "milestones": [
            {
                "name": "GraphQL Parser",
                "description": "Parse GraphQL queries into AST",
                "skills": ["Lexer", "Parser", "AST"],
                "deliverables": [
                    "GraphQL lexer/tokenizer",
                    "Recursive descent parser",
                    "Query document AST",
                    "Fragment handling",
                    "Variable definitions",
                    "Directive parsing"
                ],
                "hints": {
                    "level1": "Token types:\n```python\nclass TokenType(Enum):\n    NAME = 'NAME'\n    INT = 'INT'\n    FLOAT = 'FLOAT'\n    STRING = 'STRING'\n    LBRACE = '{'\n    RBRACE = '}'\n    LPAREN = '('\n    RPAREN = ')'\n    COLON = ':'\n    BANG = '!'\n    DOLLAR = '$'\n    AT = '@'\n    SPREAD = '...'\n```",
                    "level2": "Parse selection set:\n```python\ndef parse_selection_set(self):\n    self.expect(TokenType.LBRACE)\n    selections = []\n    while not self.check(TokenType.RBRACE):\n        if self.check(TokenType.SPREAD):\n            selections.append(self.parse_fragment())\n        else:\n            selections.append(self.parse_field())\n    self.expect(TokenType.RBRACE)\n    return SelectionSet(selections)\n```",
                    "level3": "Field with arguments:\n```python\ndef parse_field(self):\n    alias = None\n    name = self.expect(TokenType.NAME)\n    if self.check(TokenType.COLON):\n        self.advance()\n        alias = name\n        name = self.expect(TokenType.NAME)\n    \n    arguments = []\n    if self.check(TokenType.LPAREN):\n        arguments = self.parse_arguments()\n    \n    selection_set = None\n    if self.check(TokenType.LBRACE):\n        selection_set = self.parse_selection_set()\n    \n    return Field(alias, name, arguments, selection_set)\n```"
                }
            },
            {
                "name": "Schema & Type System",
                "description": "Build type system and schema representation",
                "skills": ["Type system", "Introspection", "Validation"],
                "deliverables": [
                    "Scalar types",
                    "Object types",
                    "Input types",
                    "Enum types",
                    "Interface and Union types",
                    "Introspection queries"
                ],
                "hints": {
                    "level1": "Type classes:\n```python\nclass GraphQLType:\n    pass\n\nclass GraphQLScalar(GraphQLType):\n    def __init__(self, name, serialize, parse_value, parse_literal):\n        self.name = name\n        self.serialize = serialize\n        self.parse_value = parse_value\n\nclass GraphQLObject(GraphQLType):\n    def __init__(self, name, fields):\n        self.name = name\n        self.fields = fields  # {name: GraphQLField}\n```",
                    "level2": "Schema with root types:\n```python\nclass GraphQLSchema:\n    def __init__(self, query, mutation=None, subscription=None):\n        self.query_type = query\n        self.mutation_type = mutation\n        self.subscription_type = subscription\n        self.type_map = self._build_type_map()\n    \n    def get_type(self, name):\n        return self.type_map.get(name)\n```",
                    "level3": "Introspection:\n```python\ndef add_introspection(schema):\n    schema.query_type.fields['__schema'] = GraphQLField(\n        type=__Schema,\n        resolve=lambda *_: schema\n    )\n    schema.query_type.fields['__type'] = GraphQLField(\n        type=__Type,\n        args={'name': GraphQLArgument(GraphQLString)},\n        resolve=lambda _, args, *__: schema.get_type(args['name'])\n    )\n```"
                }
            },
            {
                "name": "Query Execution",
                "description": "Execute queries against schema",
                "skills": ["Execution", "Resolvers", "Error handling"],
                "deliverables": [
                    "Field resolution",
                    "Argument coercion",
                    "List and non-null handling",
                    "Error collection",
                    "Parallel execution",
                    "Execution context"
                ],
                "hints": {
                    "level1": "Execute selection set:\n```python\nasync def execute_selection_set(self, selection_set, object_type, root_value, context):\n    results = {}\n    for selection in selection_set.selections:\n        if isinstance(selection, Field):\n            field_name = selection.alias or selection.name\n            results[field_name] = await self.execute_field(\n                selection, object_type, root_value, context\n            )\n    return results\n```",
                    "level2": "Execute field with resolver:\n```python\nasync def execute_field(self, field, parent_type, source, context):\n    field_def = parent_type.fields[field.name]\n    \n    # Coerce arguments\n    args = self.coerce_arguments(field.arguments, field_def.args)\n    \n    # Call resolver\n    resolver = field_def.resolve or default_resolver\n    try:\n        result = await maybe_await(resolver(source, args, context, field))\n    except Exception as e:\n        self.errors.append(GraphQLError(str(e), field))\n        return None\n    \n    return await self.complete_value(field_def.type, result, field, context)\n```",
                    "level3": "Complete value for type:\n```python\nasync def complete_value(self, return_type, result, field, context):\n    if isinstance(return_type, GraphQLNonNull):\n        completed = await self.complete_value(return_type.of_type, result, field, context)\n        if completed is None:\n            raise GraphQLError('Cannot return null for non-null field')\n        return completed\n    \n    if result is None:\n        return None\n    \n    if isinstance(return_type, GraphQLList):\n        return [await self.complete_value(return_type.of_type, item, field, context) for item in result]\n    \n    if isinstance(return_type, GraphQLScalar):\n        return return_type.serialize(result)\n    \n    if isinstance(return_type, GraphQLObject):\n        return await self.execute_selection_set(field.selection_set, return_type, result, context)\n```"
                }
            },
            {
                "name": "Database Schema Reflection",
                "description": "Auto-generate GraphQL schema from database",
                "skills": ["Database introspection", "Schema generation", "Relationships"],
                "deliverables": [
                    "Table to type mapping",
                    "Column to field mapping",
                    "Foreign key relationships",
                    "Primary key queries",
                    "Filter arguments",
                    "Pagination"
                ],
                "hints": {
                    "level1": "Reflect PostgreSQL tables:\n```python\nasync def reflect_tables(self, conn):\n    tables = await conn.fetch('''\n        SELECT table_name FROM information_schema.tables\n        WHERE table_schema = 'public'\n    ''')\n    \n    for table in tables:\n        columns = await conn.fetch('''\n            SELECT column_name, data_type, is_nullable\n            FROM information_schema.columns\n            WHERE table_name = $1\n        ''', table['table_name'])\n        \n        yield Table(table['table_name'], columns)\n```",
                    "level2": "Generate GraphQL type from table:\n```python\ndef table_to_graphql_type(table):\n    fields = {}\n    for col in table.columns:\n        gql_type = SQL_TO_GRAPHQL[col.data_type]\n        if col.is_nullable == 'NO':\n            gql_type = GraphQLNonNull(gql_type)\n        fields[col.column_name] = GraphQLField(gql_type)\n    \n    return GraphQLObject(pascal_case(table.name), fields)\n\nSQL_TO_GRAPHQL = {\n    'integer': GraphQLInt,\n    'bigint': GraphQLInt,\n    'text': GraphQLString,\n    'varchar': GraphQLString,\n    'boolean': GraphQLBoolean,\n    'timestamp': GraphQLDateTime,\n}\n```",
                    "level3": "Generate relationship fields:\n```python\nasync def add_relationships(self, conn, types):\n    fks = await conn.fetch('''\n        SELECT tc.table_name, kcu.column_name, \n               ccu.table_name AS foreign_table,\n               ccu.column_name AS foreign_column\n        FROM information_schema.table_constraints tc\n        JOIN information_schema.key_column_usage kcu USING (constraint_name)\n        JOIN information_schema.constraint_column_usage ccu USING (constraint_name)\n        WHERE tc.constraint_type = 'FOREIGN KEY'\n    ''')\n    \n    for fk in fks:\n        # Add object relationship (many-to-one)\n        types[fk['table_name']].fields[fk['foreign_table']] = GraphQLField(\n            types[fk['foreign_table']],\n            resolve=lambda obj, *_: fetch_by_id(fk['foreign_table'], obj[fk['column_name']])\n        )\n        # Add array relationship (one-to-many)\n        types[fk['foreign_table']].fields[fk['table_name'] + 's'] = GraphQLField(\n            GraphQLList(types[fk['table_name']]),\n            resolve=lambda obj, *_: fetch_by_fk(fk['table_name'], fk['column_name'], obj['id'])\n        )\n```"
                }
            },
            {
                "name": "Query to SQL Compilation",
                "description": "Compile GraphQL queries to efficient SQL",
                "skills": ["Query planning", "JOIN optimization", "Batching"],
                "deliverables": [
                    "Selection to SELECT mapping",
                    "Nested selection to JOINs",
                    "Filter to WHERE",
                    "Order and limit",
                    "Aggregate queries",
                    "Query batching"
                ],
                "hints": {
                    "level1": "Basic query compilation:\n```python\ndef compile_query(self, field, table):\n    columns = []\n    joins = []\n    \n    for selection in field.selection_set.selections:\n        if selection.name in table.columns:\n            columns.append(f'{table.name}.{selection.name}')\n        elif selection.name in table.relationships:\n            rel = table.relationships[selection.name]\n            joins.append(f'LEFT JOIN {rel.target_table} ON ...')\n            columns.extend(self.compile_nested(selection, rel.target_table))\n    \n    return f'SELECT {', '.join(columns)} FROM {table.name} {' '.join(joins)}'\n```",
                    "level2": "Compile filters:\n```python\ndef compile_where(self, where_arg, table):\n    clauses = []\n    params = []\n    \n    for field, condition in where_arg.items():\n        if field == '_and':\n            sub = [self.compile_where(c, table) for c in condition]\n            clauses.append(f\"({' AND '.join(s[0] for s in sub)})\")\n            params.extend(p for s in sub for p in s[1])\n        elif field == '_or':\n            sub = [self.compile_where(c, table) for c in condition]\n            clauses.append(f\"({' OR '.join(s[0] for s in sub)})\")\n        else:\n            for op, value in condition.items():\n                sql_op = {'_eq': '=', '_neq': '!=', '_gt': '>', '_lt': '<', '_like': 'LIKE'}[op]\n                clauses.append(f'{table}.{field} {sql_op} ${len(params)+1}')\n                params.append(value)\n    \n    return ' AND '.join(clauses), params\n```",
                    "level3": "Avoid N+1 with lateral join:\n```python\ndef compile_nested_lateral(self, parent, child_field):\n    return f'''\n        SELECT {parent.name}.*, \n               COALESCE(json_agg({child_field.name}.*), '[]') as {child_field.name}\n        FROM {parent.name}\n        LEFT JOIN LATERAL (\n            SELECT * FROM {child_field.table}\n            WHERE {child_field.fk} = {parent.name}.id\n        ) {child_field.name} ON true\n        GROUP BY {parent.name}.id\n    '''\n```"
                }
            }
        ]
    },
    "data-quality-framework": {
        "name": "Data Quality Framework",
        "description": "Build a data quality system with schema validation, anomaly detection, and data profiling like Great Expectations",
        "category": "Data Engineering",
        "difficulty": "intermediate",
        "estimated_hours": 45,
        "skills": [
            "Schema validation",
            "Statistical profiling",
            "Anomaly detection",
            "Rule engine",
            "Data contracts",
            "Alerting"
        ],
        "prerequisites": ["SQL", "Statistics basics", "Python"],
        "learning_outcomes": [
            "Design data validation frameworks",
            "Implement statistical data profiling",
            "Build rule-based quality checks",
            "Create data quality dashboards"
        ],
        "milestones": [
            {
                "name": "Expectation Engine",
                "description": "Define and run data quality expectations",
                "skills": ["Rule definition", "Validation logic", "Result collection"],
                "deliverables": [
                    "Expectation base class",
                    "Column expectations (not null, unique)",
                    "Value expectations (range, regex)",
                    "Result objects with metrics",
                    "Expectation suite",
                    "JSON serialization"
                ],
                "hints": {
                    "level1": "Expectation base class:\n```python\nclass Expectation:\n    def __init__(self, column=None, **kwargs):\n        self.column = column\n        self.kwargs = kwargs\n    \n    def validate(self, df) -> ExpectationResult:\n        raise NotImplementedError\n\nclass ExpectationResult:\n    def __init__(self, success, expectation, observed_value, details=None):\n        self.success = success\n        self.expectation = expectation\n        self.observed_value = observed_value\n        self.details = details or {}\n```",
                    "level2": "Common expectations:\n```python\nclass ExpectColumnNotNull(Expectation):\n    def validate(self, df):\n        null_count = df[self.column].isnull().sum()\n        total = len(df)\n        return ExpectationResult(\n            success=null_count == 0,\n            expectation=self,\n            observed_value={'null_count': null_count, 'total': total}\n        )\n\nclass ExpectColumnValuesInRange(Expectation):\n    def validate(self, df):\n        min_val, max_val = self.kwargs['min'], self.kwargs['max']\n        out_of_range = ((df[self.column] < min_val) | (df[self.column] > max_val)).sum()\n        return ExpectationResult(\n            success=out_of_range == 0,\n            expectation=self,\n            observed_value={'out_of_range': out_of_range}\n        )\n```",
                    "level3": "Expectation suite:\n```python\nclass ExpectationSuite:\n    def __init__(self, name):\n        self.name = name\n        self.expectations = []\n    \n    def add(self, expectation):\n        self.expectations.append(expectation)\n        return self\n    \n    def validate(self, df):\n        results = [exp.validate(df) for exp in self.expectations]\n        return ValidationResult(\n            success=all(r.success for r in results),\n            results=results,\n            statistics={'passed': sum(r.success for r in results), 'total': len(results)}\n        )\n    \n    def to_json(self):\n        return {'name': self.name, 'expectations': [e.to_dict() for e in self.expectations]}\n```"
                }
            },
            {
                "name": "Data Profiling",
                "description": "Automatic statistical profiling of datasets",
                "skills": ["Descriptive statistics", "Distribution analysis", "Cardinality"],
                "deliverables": [
                    "Column statistics (mean, std, min, max)",
                    "Null percentage",
                    "Cardinality and uniqueness",
                    "Value distribution",
                    "Data type inference",
                    "Profile report generation"
                ],
                "hints": {
                    "level1": "Basic column profiler:\n```python\nclass ColumnProfiler:\n    def profile(self, series):\n        return {\n            'dtype': str(series.dtype),\n            'count': len(series),\n            'null_count': series.isnull().sum(),\n            'null_pct': series.isnull().mean() * 100,\n            'unique_count': series.nunique(),\n            'unique_pct': series.nunique() / len(series) * 100\n        }\n\nclass NumericProfiler(ColumnProfiler):\n    def profile(self, series):\n        base = super().profile(series)\n        return {**base,\n            'mean': series.mean(),\n            'std': series.std(),\n            'min': series.min(),\n            'max': series.max(),\n            'percentiles': series.quantile([0.25, 0.5, 0.75]).to_dict()\n        }\n```",
                    "level2": "Value distribution:\n```python\ndef get_distribution(series, bins=20):\n    if series.dtype in ['int64', 'float64']:\n        hist, edges = np.histogram(series.dropna(), bins=bins)\n        return {'type': 'histogram', 'counts': hist.tolist(), 'edges': edges.tolist()}\n    else:\n        counts = series.value_counts().head(20)\n        return {'type': 'categorical', 'values': counts.to_dict()}\n```",
                    "level3": "Full dataset profile:\n```python\nclass DatasetProfiler:\n    def profile(self, df):\n        profiles = {}\n        for col in df.columns:\n            if pd.api.types.is_numeric_dtype(df[col]):\n                profiler = NumericProfiler()\n            elif pd.api.types.is_datetime64_dtype(df[col]):\n                profiler = DatetimeProfiler()\n            else:\n                profiler = CategoricalProfiler()\n            profiles[col] = profiler.profile(df[col])\n        \n        return DatasetProfile(\n            row_count=len(df),\n            column_count=len(df.columns),\n            columns=profiles\n        )\n```"
                }
            },
            {
                "name": "Anomaly Detection",
                "description": "Detect data anomalies and drift",
                "skills": ["Statistical tests", "Drift detection", "Outliers"],
                "deliverables": [
                    "Z-score outlier detection",
                    "IQR method",
                    "Distribution drift (KS test)",
                    "Schema drift detection",
                    "Volume anomalies",
                    "Freshness checks"
                ],
                "hints": {
                    "level1": "Outlier detection:\n```python\nclass OutlierDetector:\n    def zscore(self, series, threshold=3):\n        z = np.abs((series - series.mean()) / series.std())\n        return series[z > threshold]\n    \n    def iqr(self, series, multiplier=1.5):\n        q1, q3 = series.quantile([0.25, 0.75])\n        iqr = q3 - q1\n        lower = q1 - multiplier * iqr\n        upper = q3 + multiplier * iqr\n        return series[(series < lower) | (series > upper)]\n```",
                    "level2": "Distribution drift:\n```python\nfrom scipy import stats\n\nclass DriftDetector:\n    def __init__(self, reference_profile):\n        self.reference = reference_profile\n    \n    def detect_drift(self, current_df):\n        drift_results = {}\n        for col in current_df.columns:\n            if col in self.reference.columns:\n                ks_stat, p_value = stats.ks_2samp(\n                    self.reference[col].dropna(),\n                    current_df[col].dropna()\n                )\n                drift_results[col] = {\n                    'ks_statistic': ks_stat,\n                    'p_value': p_value,\n                    'drift_detected': p_value < 0.05\n                }\n        return drift_results\n```",
                    "level3": "Volume anomaly:\n```python\nclass VolumeMonitor:\n    def __init__(self, history_window=30):\n        self.history = []\n        self.window = history_window\n    \n    def check(self, row_count):\n        if len(self.history) < 7:\n            self.history.append(row_count)\n            return {'status': 'collecting_baseline'}\n        \n        mean = np.mean(self.history[-self.window:])\n        std = np.std(self.history[-self.window:])\n        z_score = (row_count - mean) / std if std > 0 else 0\n        \n        self.history.append(row_count)\n        return {\n            'row_count': row_count,\n            'expected': mean,\n            'z_score': z_score,\n            'anomaly': abs(z_score) > 3\n        }\n```"
                }
            },
            {
                "name": "Data Contracts",
                "description": "Schema contracts and versioning",
                "skills": ["Schema definition", "Contract validation", "Versioning"],
                "deliverables": [
                    "Schema contract YAML format",
                    "Contract validation",
                    "Breaking change detection",
                    "Contract versioning",
                    "Producer/consumer registration",
                    "Contract testing"
                ],
                "hints": {
                    "level1": "Contract definition:\n```yaml\n# data_contract.yaml\nname: user_events\nversion: 1.0.0\nowner: data-team\n\nschema:\n  columns:\n    - name: user_id\n      type: string\n      required: true\n    - name: event_type\n      type: string\n      required: true\n      allowed_values: [click, view, purchase]\n    - name: timestamp\n      type: timestamp\n      required: true\n    - name: amount\n      type: decimal\n      required: false\n\nquality:\n  - expect_column_values_not_null:\n      column: user_id\n  - expect_column_values_in_set:\n      column: event_type\n      values: [click, view, purchase]\n```",
                    "level2": "Contract validator:\n```python\nclass ContractValidator:\n    def __init__(self, contract_path):\n        with open(contract_path) as f:\n            self.contract = yaml.safe_load(f)\n    \n    def validate_schema(self, df):\n        errors = []\n        for col_spec in self.contract['schema']['columns']:\n            if col_spec['name'] not in df.columns:\n                if col_spec.get('required', False):\n                    errors.append(f\"Missing required column: {col_spec['name']}\")\n            else:\n                actual_type = str(df[col_spec['name']].dtype)\n                if not self.types_compatible(actual_type, col_spec['type']):\n                    errors.append(f\"Type mismatch for {col_spec['name']}\")\n        return errors\n```",
                    "level3": "Breaking change detection:\n```python\ndef detect_breaking_changes(old_contract, new_contract):\n    breaking = []\n    \n    old_cols = {c['name']: c for c in old_contract['schema']['columns']}\n    new_cols = {c['name']: c for c in new_contract['schema']['columns']}\n    \n    for name, spec in old_cols.items():\n        if name not in new_cols:\n            breaking.append(f'Column removed: {name}')\n        elif new_cols[name]['type'] != spec['type']:\n            breaking.append(f'Type changed: {name}')\n        elif not spec.get('required') and new_cols[name].get('required'):\n            breaking.append(f'Column became required: {name}')\n    \n    return breaking\n```"
                }
            }
        ]
    },
    "stream-processing-engine": {
        "name": "Stream Processing Engine",
        "description": "Build a real-time stream processing engine with windowing, state management, and exactly-once semantics like Flink",
        "category": "Data Engineering",
        "difficulty": "expert",
        "estimated_hours": 100,
        "skills": [
            "Event time processing",
            "Windowing",
            "State management",
            "Checkpointing",
            "Watermarks",
            "Exactly-once semantics"
        ],
        "prerequisites": ["Distributed systems", "Message queues", "State machines"],
        "learning_outcomes": [
            "Understand stream processing fundamentals",
            "Implement windowing and watermarks",
            "Build fault-tolerant stateful processing",
            "Design exactly-once delivery systems"
        ],
        "milestones": [
            {
                "name": "Stream Abstraction",
                "description": "Core stream data structures and operators",
                "skills": ["Stream API", "Operators", "Transformations"],
                "deliverables": [
                    "DataStream class",
                    "Map, filter, flatMap operators",
                    "KeyBy for partitioning",
                    "Operator chaining",
                    "Source and sink interfaces",
                    "Execution graph"
                ],
                "hints": {
                    "level1": "DataStream API:\n```python\nclass DataStream:\n    def __init__(self, source, operators=None):\n        self.source = source\n        self.operators = operators or []\n    \n    def map(self, fn):\n        return DataStream(self.source, self.operators + [MapOperator(fn)])\n    \n    def filter(self, fn):\n        return DataStream(self.source, self.operators + [FilterOperator(fn)])\n    \n    def key_by(self, key_fn):\n        return KeyedStream(self.source, self.operators, key_fn)\n```",
                    "level2": "Operators:\n```python\nclass Operator:\n    def process(self, element):\n        raise NotImplementedError\n\nclass MapOperator(Operator):\n    def __init__(self, fn):\n        self.fn = fn\n    \n    def process(self, element):\n        yield self.fn(element)\n\nclass FilterOperator(Operator):\n    def __init__(self, predicate):\n        self.predicate = predicate\n    \n    def process(self, element):\n        if self.predicate(element):\n            yield element\n```",
                    "level3": "Execution graph:\n```python\nclass StreamExecutionEnvironment:\n    def __init__(self):\n        self.sources = []\n    \n    def add_source(self, source):\n        stream = DataStream(source)\n        self.sources.append(stream)\n        return stream\n    \n    def execute(self):\n        graph = self.build_graph()\n        for task in graph.tasks:\n            self.submit_task(task)\n```"
                }
            },
            {
                "name": "Windowing",
                "description": "Time-based windowing for aggregations",
                "skills": ["Window types", "Triggers", "Evictors"],
                "deliverables": [
                    "Tumbling windows",
                    "Sliding windows",
                    "Session windows",
                    "Window assigners",
                    "Triggers (count, time)",
                    "Late data handling"
                ],
                "hints": {
                    "level1": "Window types:\n```python\nclass TumblingWindow:\n    def __init__(self, size_ms):\n        self.size = size_ms\n    \n    def assign(self, timestamp):\n        start = timestamp - (timestamp % self.size)\n        return Window(start, start + self.size)\n\nclass SlidingWindow:\n    def __init__(self, size_ms, slide_ms):\n        self.size = size_ms\n        self.slide = slide_ms\n    \n    def assign(self, timestamp):\n        windows = []\n        start = timestamp - (timestamp % self.slide)\n        while start > timestamp - self.size:\n            windows.append(Window(start, start + self.size))\n            start -= self.slide\n        return windows\n```",
                    "level2": "Windowed stream:\n```python\nclass WindowedStream:\n    def __init__(self, keyed_stream, window_assigner):\n        self.keyed_stream = keyed_stream\n        self.window_assigner = window_assigner\n        self.window_buffers = defaultdict(lambda: defaultdict(list))\n    \n    def process(self, element):\n        key = self.keyed_stream.key_fn(element)\n        windows = self.window_assigner.assign(element.timestamp)\n        for window in windows:\n            self.window_buffers[key][window].append(element)\n    \n    def reduce(self, fn):\n        return WindowReduceStream(self, fn)\n```",
                    "level3": "Triggers:\n```python\nclass Trigger:\n    def on_element(self, element, window, ctx):\n        raise NotImplementedError\n    \n    def on_processing_time(self, time, window, ctx):\n        raise NotImplementedError\n\nclass EventTimeTrigger(Trigger):\n    def on_element(self, element, window, ctx):\n        if ctx.current_watermark >= window.end:\n            return TriggerResult.FIRE_AND_PURGE\n        ctx.register_event_time_timer(window.end)\n        return TriggerResult.CONTINUE\n```"
                }
            },
            {
                "name": "Event Time & Watermarks",
                "description": "Handle out-of-order events with watermarks",
                "skills": ["Event time", "Watermark generation", "Lateness"],
                "deliverables": [
                    "Event time extraction",
                    "Watermark generation strategies",
                    "Bounded out-of-orderness",
                    "Watermark propagation",
                    "Idle source handling",
                    "Allowed lateness"
                ],
                "hints": {
                    "level1": "Watermark generator:\n```python\nclass WatermarkGenerator:\n    def __init__(self, max_out_of_orderness):\n        self.max_ooo = max_out_of_orderness\n        self.max_timestamp = 0\n    \n    def on_event(self, event, timestamp):\n        self.max_timestamp = max(self.max_timestamp, timestamp)\n    \n    def get_watermark(self):\n        return Watermark(self.max_timestamp - self.max_ooo)\n```",
                    "level2": "Watermark propagation:\n```python\nclass WatermarkCoordinator:\n    def __init__(self, sources):\n        self.source_watermarks = {s: 0 for s in sources}\n    \n    def update(self, source, watermark):\n        self.source_watermarks[source] = watermark\n        # Min watermark across all sources\n        return min(self.source_watermarks.values())\n    \n    def handle_idle_source(self, source, timeout_ms):\n        # Mark source as idle, exclude from min calculation\n        if time.time() - self.last_event[source] > timeout_ms:\n            self.source_watermarks[source] = float('inf')\n```",
                    "level3": "Late data handling:\n```python\nclass WindowOperator:\n    def __init__(self, allowed_lateness=0):\n        self.allowed_lateness = allowed_lateness\n        self.late_output = SideOutput('late')\n    \n    def process_watermark(self, watermark):\n        # Fire windows that are complete\n        for window in self.pending_windows:\n            if watermark.timestamp >= window.end:\n                self.fire_window(window)\n        \n        # Clean up windows past allowed lateness\n        cleanup_time = watermark.timestamp - self.allowed_lateness\n        self.cleanup_windows_before(cleanup_time)\n    \n    def process_element(self, element):\n        windows = self.assigner.assign(element.timestamp)\n        for window in windows:\n            if self.is_late(element, window):\n                self.late_output.emit(element)\n            else:\n                self.add_to_window(element, window)\n```"
                }
            },
            {
                "name": "Stateful Processing",
                "description": "Manage operator state with checkpointing",
                "skills": ["State backends", "Keyed state", "Checkpointing"],
                "deliverables": [
                    "ValueState, ListState, MapState",
                    "State backend interface",
                    "RocksDB state backend",
                    "Checkpoint barriers",
                    "Async checkpointing",
                    "State recovery"
                ],
                "hints": {
                    "level1": "State abstractions:\n```python\nclass ValueState:\n    def __init__(self, backend, key, name):\n        self.backend = backend\n        self.key = key\n        self.name = name\n    \n    def value(self):\n        return self.backend.get(self.key, self.name)\n    \n    def update(self, value):\n        self.backend.put(self.key, self.name, value)\n\nclass KeyedStateBackend:\n    def __init__(self):\n        self.state = defaultdict(dict)  # key -> {name -> value}\n    \n    def get(self, key, name):\n        return self.state[key].get(name)\n    \n    def put(self, key, name, value):\n        self.state[key][name] = value\n```",
                    "level2": "Checkpoint barriers:\n```python\nclass CheckpointBarrier:\n    def __init__(self, checkpoint_id, timestamp):\n        self.checkpoint_id = checkpoint_id\n        self.timestamp = timestamp\n\nclass BarrierHandler:\n    def __init__(self, num_inputs):\n        self.pending_barriers = {}\n        self.num_inputs = num_inputs\n    \n    def process_barrier(self, barrier, input_channel):\n        cp_id = barrier.checkpoint_id\n        if cp_id not in self.pending_barriers:\n            self.pending_barriers[cp_id] = set()\n        \n        self.pending_barriers[cp_id].add(input_channel)\n        \n        if len(self.pending_barriers[cp_id]) == self.num_inputs:\n            # All barriers received - trigger checkpoint\n            self.trigger_checkpoint(cp_id)\n            del self.pending_barriers[cp_id]\n```",
                    "level3": "RocksDB state backend:\n```python\nimport rocksdb\n\nclass RocksDBStateBackend:\n    def __init__(self, path):\n        self.db = rocksdb.DB(path, rocksdb.Options(create_if_missing=True))\n    \n    def get(self, key, name):\n        full_key = f'{key}:{name}'.encode()\n        value = self.db.get(full_key)\n        return pickle.loads(value) if value else None\n    \n    def put(self, key, name, value):\n        full_key = f'{key}:{name}'.encode()\n        self.db.put(full_key, pickle.dumps(value))\n    \n    def snapshot(self, checkpoint_path):\n        checkpoint = rocksdb.Checkpoint(self.db)\n        checkpoint.create_checkpoint(checkpoint_path)\n```"
                }
            },
            {
                "name": "Exactly-Once Semantics",
                "description": "Guarantee exactly-once processing",
                "skills": ["Two-phase commit", "Idempotent sinks", "Transaction coordination"],
                "deliverables": [
                    "Transactional sources",
                    "Two-phase commit sinks",
                    "Checkpoint completion callbacks",
                    "Transaction coordinator",
                    "Abort and recovery",
                    "End-to-end exactly-once"
                ],
                "hints": {
                    "level1": "Two-phase commit sink:\n```python\nclass TwoPhaseCommitSink:\n    def __init__(self):\n        self.pending_transactions = {}\n    \n    def invoke(self, value, context):\n        # Write to pending transaction\n        txn = self.current_transaction()\n        txn.write(value)\n    \n    def snapshot_state(self, checkpoint_id):\n        # Pre-commit: flush but don't commit\n        txn = self.current_transaction()\n        txn.pre_commit()\n        self.pending_transactions[checkpoint_id] = txn\n        self.begin_new_transaction()\n    \n    def notify_checkpoint_complete(self, checkpoint_id):\n        # Commit on checkpoint success\n        txn = self.pending_transactions.pop(checkpoint_id)\n        txn.commit()\n```",
                    "level2": "Kafka exactly-once:\n```python\nclass KafkaExactlyOnceSink(TwoPhaseCommitSink):\n    def __init__(self, bootstrap_servers):\n        self.producer = KafkaProducer(\n            bootstrap_servers=bootstrap_servers,\n            transactional_id='my-transactional-id'\n        )\n        self.producer.init_transactions()\n    \n    def begin_transaction(self):\n        self.producer.begin_transaction()\n    \n    def pre_commit(self):\n        self.producer.flush()\n    \n    def commit(self):\n        self.producer.commit_transaction()\n    \n    def abort(self):\n        self.producer.abort_transaction()\n```",
                    "level3": "Transaction coordinator:\n```python\nclass TransactionCoordinator:\n    def __init__(self, sinks):\n        self.sinks = sinks\n    \n    def on_checkpoint_complete(self, checkpoint_id):\n        # Phase 2: Commit all sinks\n        for sink in self.sinks:\n            try:\n                sink.commit(checkpoint_id)\n            except Exception as e:\n                # If any commit fails, need to handle recovery\n                self.handle_commit_failure(checkpoint_id, sink, e)\n    \n    def on_checkpoint_abort(self, checkpoint_id):\n        for sink in self.sinks:\n            sink.abort(checkpoint_id)\n```"
                }
            }
        ]
    },
    "data-lakehouse": {
        "name": "Data Lakehouse",
        "description": "Build a data lakehouse with ACID transactions on object storage like Delta Lake",
        "category": "Data Engineering",
        "difficulty": "expert",
        "estimated_hours": 90,
        "skills": [
            "ACID transactions",
            "Parquet format",
            "Transaction log",
            "Time travel",
            "Schema evolution",
            "Compaction"
        ],
        "prerequisites": ["Distributed storage", "File formats", "Transaction concepts"],
        "learning_outcomes": [
            "Understand lakehouse architecture",
            "Implement ACID on object storage",
            "Build time travel and versioning",
            "Design schema evolution systems"
        ],
        "milestones": [
            {
                "name": "Transaction Log",
                "description": "Implement commit log for ACID",
                "skills": ["Log structure", "Atomic commits", "JSON format"],
                "deliverables": [
                    "Log entry format",
                    "Atomic commit protocol",
                    "Log compaction (checkpoints)",
                    "Log replay on read",
                    "Version numbering",
                    "Concurrent write handling"
                ],
                "hints": {
                    "level1": "Log entry format:\n```python\n@dataclass\nclass LogEntry:\n    version: int\n    timestamp: int\n    actions: List[Action]\n\n@dataclass\nclass AddFile:\n    path: str\n    partition_values: dict\n    size: int\n    stats: dict  # min/max values, row count\n\n@dataclass\nclass RemoveFile:\n    path: str\n    deletion_timestamp: int\n```",
                    "level2": "Atomic commit:\n```python\nclass TransactionLog:\n    def __init__(self, table_path, storage):\n        self.table_path = table_path\n        self.storage = storage\n        self.log_path = f'{table_path}/_delta_log'\n    \n    def commit(self, version, actions):\n        entry = LogEntry(version, time.time(), actions)\n        log_file = f'{self.log_path}/{version:020d}.json'\n        \n        # Atomic write: write to temp, then rename\n        temp_file = f'{log_file}.tmp.{uuid4()}'\n        self.storage.write(temp_file, json.dumps(entry))\n        \n        try:\n            self.storage.rename(temp_file, log_file)\n        except FileExistsError:\n            self.storage.delete(temp_file)\n            raise ConcurrentModificationError()\n```",
                    "level3": "Checkpoint for faster reads:\n```python\ndef create_checkpoint(self, version):\n    # Replay log to get current state\n    state = self.replay_log(0, version)\n    \n    # Write checkpoint as parquet\n    checkpoint_path = f'{self.log_path}/{version:020d}.checkpoint.parquet'\n    df = pd.DataFrame([asdict(f) for f in state.files])\n    df.to_parquet(checkpoint_path)\n    \n    # Update _last_checkpoint\n    self.storage.write(\n        f'{self.log_path}/_last_checkpoint',\n        json.dumps({'version': version})\n    )\n```"
                }
            },
            {
                "name": "Read Path",
                "description": "Read data with snapshot isolation",
                "skills": ["Snapshot isolation", "File listing", "Predicate pushdown"],
                "deliverables": [
                    "Snapshot construction",
                    "Active file listing",
                    "Partition pruning",
                    "File statistics for pruning",
                    "Predicate pushdown",
                    "Time travel queries"
                ],
                "hints": {
                    "level1": "Get snapshot at version:\n```python\nclass Snapshot:\n    def __init__(self, version, files):\n        self.version = version\n        self.files = files  # Set of active AddFile\n    \n    @classmethod\n    def from_log(cls, log, version=None):\n        if version is None:\n            version = log.latest_version()\n        \n        state = log.replay_log(0, version)\n        return cls(version, state.files)\n    \n    def scan(self, predicate=None):\n        files = self.files\n        if predicate:\n            files = [f for f in files if self.might_contain(f, predicate)]\n        return files\n```",
                    "level2": "File pruning with stats:\n```python\ndef might_contain(self, file, predicate):\n    # Use min/max stats for pruning\n    stats = file.stats\n    \n    for col, op, value in predicate.conditions:\n        if col in stats.get('min', {}):\n            if op == '>' and stats['max'][col] <= value:\n                return False\n            if op == '<' and stats['min'][col] >= value:\n                return False\n            if op == '=' and (stats['min'][col] > value or stats['max'][col] < value):\n                return False\n    return True\n```",
                    "level3": "Time travel:\n```python\nclass DeltaTable:\n    def as_of_version(self, version):\n        return Snapshot.from_log(self.log, version)\n    \n    def as_of_timestamp(self, timestamp):\n        # Find version at timestamp\n        version = self.log.version_at_timestamp(timestamp)\n        return self.as_of_version(version)\n    \n    def history(self, limit=None):\n        entries = []\n        for v in range(self.log.latest_version(), -1, -1):\n            entry = self.log.read_entry(v)\n            entries.append({\n                'version': v,\n                'timestamp': entry.timestamp,\n                'operation': entry.operation\n            })\n            if limit and len(entries) >= limit:\n                break\n        return entries\n```"
                }
            },
            {
                "name": "Write Path",
                "description": "Write data with ACID guarantees",
                "skills": ["Write transactions", "Merge operations", "Conflict detection"],
                "deliverables": [
                    "Insert operation",
                    "Update with predicates",
                    "Delete with predicates",
                    "Merge (upsert) operation",
                    "Optimistic concurrency control",
                    "Conflict resolution"
                ],
                "hints": {
                    "level1": "Insert operation:\n```python\nclass DeltaTable:\n    def insert(self, df):\n        # Write parquet files\n        file_path = f'{self.table_path}/data/{uuid4()}.parquet'\n        df.to_parquet(file_path)\n        \n        # Compute stats\n        stats = self.compute_stats(df)\n        \n        # Commit to log\n        action = AddFile(\n            path=file_path,\n            partition_values={},\n            size=os.path.getsize(file_path),\n            stats=stats\n        )\n        self.commit([action])\n```",
                    "level2": "Update with conflict detection:\n```python\ndef update(self, predicate, updates):\n    snapshot = self.snapshot()\n    \n    # Find files that might match\n    matching_files = snapshot.scan(predicate)\n    \n    actions = []\n    for file in matching_files:\n        df = pd.read_parquet(file.path)\n        mask = predicate.evaluate(df)\n        \n        if mask.any():\n            # Apply updates\n            for col, value in updates.items():\n                df.loc[mask, col] = value\n            \n            # Write new file\n            new_path = f'{self.table_path}/data/{uuid4()}.parquet'\n            df.to_parquet(new_path)\n            \n            actions.append(RemoveFile(file.path, time.time()))\n            actions.append(AddFile(new_path, {}, os.path.getsize(new_path), self.compute_stats(df)))\n    \n    # Commit with read files for conflict detection\n    self.commit(actions, read_files=[f.path for f in matching_files])\n```",
                    "level3": "Merge (upsert):\n```python\ndef merge(self, source_df, condition, when_matched=None, when_not_matched=None):\n    target_snapshot = self.snapshot()\n    \n    # Read all potentially matching files\n    target_df = self.read_files(target_snapshot.files)\n    \n    # Join source and target\n    merged = source_df.merge(target_df, on=condition.keys, how='outer', indicator=True)\n    \n    # Apply matched updates\n    matched_mask = merged['_merge'] == 'both'\n    if when_matched:\n        for col, expr in when_matched.items():\n            merged.loc[matched_mask, col] = expr.evaluate(merged[matched_mask])\n    \n    # Apply not matched inserts\n    not_matched_mask = merged['_merge'] == 'left_only'\n    if when_not_matched:\n        inserts = merged[not_matched_mask][when_not_matched.columns]\n    \n    # Write results and commit\n    ...\n```"
                }
            },
            {
                "name": "Schema Evolution",
                "description": "Handle schema changes safely",
                "skills": ["Schema compatibility", "Column mapping", "Type widening"],
                "deliverables": [
                    "Schema storage in metadata",
                    "Add column operation",
                    "Rename column",
                    "Type widening rules",
                    "Schema enforcement modes",
                    "Column mapping"
                ],
                "hints": {
                    "level1": "Schema in metadata:\n```python\n@dataclass\nclass Metadata:\n    schema: Schema\n    partition_columns: List[str]\n    configuration: dict\n\n@dataclass\nclass Schema:\n    fields: List[Field]\n\n@dataclass \nclass Field:\n    name: str\n    type: str\n    nullable: bool\n    metadata: dict = None\n```",
                    "level2": "Schema evolution:\n```python\nclass SchemaEvolution:\n    COMPATIBLE_WIDENING = {\n        ('int', 'long'): True,\n        ('float', 'double'): True,\n        ('date', 'timestamp'): True,\n    }\n    \n    def can_evolve(self, current, new):\n        # Adding nullable column is safe\n        for field in new.fields:\n            if field.name not in [f.name for f in current.fields]:\n                if not field.nullable:\n                    return False, 'Cannot add non-nullable column'\n        \n        # Check type compatibility\n        for curr_field in current.fields:\n            new_field = new.get_field(curr_field.name)\n            if new_field and curr_field.type != new_field.type:\n                if (curr_field.type, new_field.type) not in self.COMPATIBLE_WIDENING:\n                    return False, f'Incompatible type change: {curr_field.name}'\n        \n        return True, None\n```",
                    "level3": "Column mapping for renames:\n```python\nclass ColumnMapping:\n    def __init__(self):\n        self.id_to_name = {}  # Stable ID -> current name\n        self.name_to_id = {}  # Current name -> stable ID\n        self.next_id = 0\n    \n    def add_column(self, name):\n        col_id = self.next_id\n        self.next_id += 1\n        self.id_to_name[col_id] = name\n        self.name_to_id[name] = col_id\n        return col_id\n    \n    def rename_column(self, old_name, new_name):\n        col_id = self.name_to_id[old_name]\n        del self.name_to_id[old_name]\n        self.name_to_id[new_name] = col_id\n        self.id_to_name[col_id] = new_name\n\n# Files store column IDs, not names\n# Read maps IDs to current names\n```"
                }
            },
            {
                "name": "Compaction & Optimization",
                "description": "Optimize table layout and performance",
                "skills": ["File compaction", "Z-ordering", "Vacuum"],
                "deliverables": [
                    "Small file compaction",
                    "Bin-packing",
                    "Z-order clustering",
                    "Vacuum (delete old files)",
                    "Optimize command",
                    "Auto-compaction"
                ],
                "hints": {
                    "level1": "Compact small files:\n```python\ndef compact(self, target_size_mb=128):\n    snapshot = self.snapshot()\n    target_size = target_size_mb * 1024 * 1024\n    \n    # Group small files\n    small_files = [f for f in snapshot.files if f.size < target_size * 0.75]\n    \n    # Bin pack into groups\n    groups = []\n    current_group = []\n    current_size = 0\n    \n    for f in sorted(small_files, key=lambda x: x.size):\n        if current_size + f.size > target_size:\n            groups.append(current_group)\n            current_group = [f]\n            current_size = f.size\n        else:\n            current_group.append(f)\n            current_size += f.size\n    \n    if current_group:\n        groups.append(current_group)\n    \n    # Rewrite each group\n    for group in groups:\n        self.rewrite_files(group)\n```",
                    "level2": "Z-order clustering:\n```python\ndef zorder(self, columns):\n    snapshot = self.snapshot()\n    \n    # Read all data\n    df = self.read_files(snapshot.files)\n    \n    # Compute Z-order values\n    z_values = self.compute_z_values(df, columns)\n    df['_z_order'] = z_values\n    \n    # Sort and write in chunks\n    df_sorted = df.sort_values('_z_order').drop('_z_order', axis=1)\n    \n    new_files = self.write_in_chunks(df_sorted, chunk_size=128*1024*1024)\n    \n    # Commit: remove old, add new\n    actions = [RemoveFile(f.path, time.time()) for f in snapshot.files]\n    actions += [AddFile(...) for f in new_files]\n    self.commit(actions)\n```",
                    "level3": "Vacuum old files:\n```python\ndef vacuum(self, retention_hours=168):  # 7 days default\n    cutoff = time.time() - retention_hours * 3600\n    \n    # Get all RemoveFile actions older than cutoff\n    files_to_delete = []\n    for entry in self.log.entries():\n        for action in entry.actions:\n            if isinstance(action, RemoveFile):\n                if action.deletion_timestamp < cutoff:\n                    files_to_delete.append(action.path)\n    \n    # Check not referenced by any active version\n    for version in self.log.versions_after(cutoff):\n        snapshot = self.snapshot(version)\n        for f in snapshot.files:\n            if f.path in files_to_delete:\n                files_to_delete.remove(f.path)\n    \n    # Delete files\n    for path in files_to_delete:\n        self.storage.delete(path)\n    \n    return len(files_to_delete)\n```"
                }
            }
        ]
    },
    "gitops-deployment": {
        "name": "GitOps Deployment System",
        "description": "Build a GitOps deployment system with Git as source of truth, like ArgoCD",
        "category": "Cloud Native",
        "difficulty": "advanced",
        "estimated_hours": 55,
        "skills": [
            "Git operations",
            "Kubernetes API",
            "Reconciliation",
            "Diff detection",
            "Rollback",
            "Health checks"
        ],
        "prerequisites": ["Kubernetes basics", "Git", "YAML/Helm"],
        "learning_outcomes": [
            "Understand GitOps principles",
            "Implement Git-driven deployments",
            "Build sync and health monitoring",
            "Handle rollback and recovery"
        ],
        "milestones": [
            {
                "name": "Git Repository Sync",
                "description": "Clone and sync Git repositories",
                "skills": ["Git operations", "Polling/webhooks", "Credential management"],
                "deliverables": [
                    "Repository cloning",
                    "Branch/tag tracking",
                    "Polling for changes",
                    "Webhook receiver",
                    "SSH/HTTPS credentials",
                    "Repository caching"
                ],
                "hints": {
                    "level1": "Git sync:\n```python\nimport git\n\nclass GitRepoSync:\n    def __init__(self, url, branch, path, credentials=None):\n        self.url = url\n        self.branch = branch\n        self.path = path\n        self.credentials = credentials\n    \n    def sync(self):\n        if os.path.exists(self.path):\n            repo = git.Repo(self.path)\n            repo.remotes.origin.pull()\n        else:\n            repo = git.Repo.clone_from(self.url, self.path, branch=self.branch)\n        return repo.head.commit.hexsha\n```",
                    "level2": "Webhook handler:\n```python\n@app.post('/webhook/github')\nasync def github_webhook(request: Request):\n    payload = await request.json()\n    \n    if request.headers.get('X-GitHub-Event') == 'push':\n        repo_url = payload['repository']['clone_url']\n        branch = payload['ref'].split('/')[-1]\n        commit = payload['after']\n        \n        # Trigger sync for matching applications\n        apps = await get_apps_for_repo(repo_url, branch)\n        for app in apps:\n            await trigger_sync(app, commit)\n    \n    return {'status': 'ok'}\n```",
                    "level3": "Credential management:\n```python\nclass CredentialStore:\n    def __init__(self, k8s_client):\n        self.k8s = k8s_client\n    \n    async def get_credentials(self, secret_name, namespace):\n        secret = await self.k8s.read_namespaced_secret(secret_name, namespace)\n        \n        if 'sshPrivateKey' in secret.data:\n            return SSHCredentials(\n                private_key=base64.b64decode(secret.data['sshPrivateKey'])\n            )\n        elif 'username' in secret.data:\n            return HTTPSCredentials(\n                username=base64.b64decode(secret.data['username']).decode(),\n                password=base64.b64decode(secret.data['password']).decode()\n            )\n```"
                }
            },
            {
                "name": "Manifest Generation",
                "description": "Generate Kubernetes manifests from source",
                "skills": ["YAML processing", "Helm", "Kustomize"],
                "deliverables": [
                    "Plain YAML reading",
                    "Helm template rendering",
                    "Kustomize build",
                    "Parameter overrides",
                    "Environment-specific values",
                    "Manifest validation"
                ],
                "hints": {
                    "level1": "Manifest generator interface:\n```python\nclass ManifestGenerator:\n    def generate(self, source_path, params) -> List[dict]:\n        raise NotImplementedError\n\nclass PlainYAMLGenerator(ManifestGenerator):\n    def generate(self, source_path, params):\n        manifests = []\n        for file in glob.glob(f'{source_path}/**/*.yaml', recursive=True):\n            with open(file) as f:\n                for doc in yaml.safe_load_all(f):\n                    if doc:\n                        manifests.append(doc)\n        return manifests\n```",
                    "level2": "Helm template:\n```python\nclass HelmGenerator(ManifestGenerator):\n    def generate(self, chart_path, params):\n        values_file = self.write_values(params.get('values', {}))\n        \n        cmd = [\n            'helm', 'template',\n            params.get('release_name', 'release'),\n            chart_path,\n            '-f', values_file,\n            '--namespace', params.get('namespace', 'default')\n        ]\n        \n        result = subprocess.run(cmd, capture_output=True, text=True)\n        if result.returncode != 0:\n            raise HelmError(result.stderr)\n        \n        return list(yaml.safe_load_all(result.stdout))\n```",
                    "level3": "Kustomize:\n```python\nclass KustomizeGenerator(ManifestGenerator):\n    def generate(self, kustomize_path, params):\n        # Apply parameter overrides\n        if params.get('images'):\n            for image in params['images']:\n                subprocess.run([\n                    'kustomize', 'edit', 'set', 'image',\n                    f\"{image['name']}={image['newName']}:{image['newTag']}\"\n                ], cwd=kustomize_path)\n        \n        result = subprocess.run(\n            ['kustomize', 'build', kustomize_path],\n            capture_output=True, text=True\n        )\n        return list(yaml.safe_load_all(result.stdout))\n```"
                }
            },
            {
                "name": "Sync & Reconciliation",
                "description": "Apply manifests and reconcile state",
                "skills": ["K8s apply", "Diff detection", "Pruning"],
                "deliverables": [
                    "Manifest diff calculation",
                    "Apply with server-side apply",
                    "Resource pruning",
                    "Sync waves/hooks",
                    "Selective sync",
                    "Dry-run mode"
                ],
                "hints": {
                    "level1": "Sync application:\n```python\nclass Syncer:\n    def __init__(self, k8s_client):\n        self.k8s = k8s_client\n    \n    async def sync(self, app, manifests):\n        results = []\n        for manifest in manifests:\n            result = await self.apply_manifest(manifest)\n            results.append(result)\n        \n        if app.spec.sync_policy.prune:\n            await self.prune_orphans(app, manifests)\n        \n        return SyncResult(results)\n    \n    async def apply_manifest(self, manifest):\n        api = self.get_api_for_kind(manifest['kind'])\n        try:\n            await api.patch(\n                name=manifest['metadata']['name'],\n                namespace=manifest['metadata'].get('namespace'),\n                body=manifest,\n                field_manager='gitops-controller'\n            )\n            return ApplyResult(manifest, 'synced')\n        except Exception as e:\n            return ApplyResult(manifest, 'failed', str(e))\n```",
                    "level2": "Diff calculation:\n```python\ndef calculate_diff(self, desired, live):\n    diffs = []\n    \n    desired_map = {self.resource_key(m): m for m in desired}\n    live_map = {self.resource_key(m): m for m in live}\n    \n    for key, manifest in desired_map.items():\n        if key not in live_map:\n            diffs.append(Diff(manifest, None, 'add'))\n        else:\n            live_manifest = live_map[key]\n            if not self.equal(manifest, live_manifest):\n                diffs.append(Diff(manifest, live_manifest, 'modify'))\n    \n    for key, manifest in live_map.items():\n        if key not in desired_map:\n            diffs.append(Diff(None, manifest, 'delete'))\n    \n    return diffs\n```",
                    "level3": "Sync waves:\n```python\ndef group_by_wave(self, manifests):\n    waves = defaultdict(list)\n    for m in manifests:\n        wave = int(m.get('metadata', {}).get('annotations', {}).get('argocd.io/sync-wave', '0'))\n        waves[wave].append(m)\n    return sorted(waves.items())\n\nasync def sync_with_waves(self, app, manifests):\n    for wave_num, wave_manifests in self.group_by_wave(manifests):\n        # Apply all in wave\n        results = await asyncio.gather(*[\n            self.apply_manifest(m) for m in wave_manifests\n        ])\n        \n        # Wait for health before next wave\n        await self.wait_for_health(wave_manifests)\n```"
                }
            },
            {
                "name": "Health Assessment",
                "description": "Monitor application and resource health",
                "skills": ["Health checks", "Status aggregation", "Custom health"],
                "deliverables": [
                    "Built-in health checks per kind",
                    "Deployment rollout status",
                    "Custom health scripts",
                    "Application health aggregation",
                    "Degraded vs healthy vs progressing",
                    "Health history"
                ],
                "hints": {
                    "level1": "Health check per resource type:\n```python\nclass HealthChecker:\n    def check(self, resource) -> HealthStatus:\n        kind = resource['kind']\n        checker = getattr(self, f'check_{kind.lower()}', self.check_default)\n        return checker(resource)\n    \n    def check_deployment(self, deployment):\n        status = deployment.get('status', {})\n        replicas = status.get('replicas', 0)\n        ready = status.get('readyReplicas', 0)\n        updated = status.get('updatedReplicas', 0)\n        \n        if ready == replicas == updated:\n            return HealthStatus.HEALTHY\n        elif updated < replicas:\n            return HealthStatus.PROGRESSING\n        else:\n            return HealthStatus.DEGRADED\n```",
                    "level2": "Aggregate application health:\n```python\ndef aggregate_health(self, resources):\n    statuses = [self.check(r) for r in resources]\n    \n    if all(s == HealthStatus.HEALTHY for s in statuses):\n        return HealthStatus.HEALTHY\n    elif any(s == HealthStatus.DEGRADED for s in statuses):\n        return HealthStatus.DEGRADED\n    elif any(s == HealthStatus.PROGRESSING for s in statuses):\n        return HealthStatus.PROGRESSING\n    else:\n        return HealthStatus.UNKNOWN\n```",
                    "level3": "Custom health check (Lua script):\n```python\nimport lupa\n\nclass CustomHealthChecker:\n    def __init__(self):\n        self.lua = lupa.LuaRuntime()\n    \n    def check_with_script(self, resource, script):\n        func = self.lua.eval(f'''\n            function(obj)\n                {script}\n            end\n        ''')\n        result = func(resource)\n        return HealthStatus(result['status']), result.get('message')\n\n# Example Lua script:\n# if obj.status.phase == \"Running\" then\n#     return {status = \"Healthy\"}\n# else\n#     return {status = \"Progressing\", message = \"Waiting for pod\"}\n# end\n```"
                }
            },
            {
                "name": "Rollback & History",
                "description": "Track history and enable rollback",
                "skills": ["Version tracking", "Rollback", "Audit"],
                "deliverables": [
                    "Revision history storage",
                    "Rollback to revision",
                    "Auto-rollback on failure",
                    "Deployment annotations",
                    "Audit logging",
                    "Sync status tracking"
                ],
                "hints": {
                    "level1": "Revision history:\n```python\n@dataclass\nclass Revision:\n    id: int\n    commit_sha: str\n    deployed_at: datetime\n    manifests_hash: str\n    sync_status: str\n\nclass RevisionHistory:\n    def __init__(self, app_name, storage):\n        self.app_name = app_name\n        self.storage = storage\n    \n    def record(self, commit_sha, manifests):\n        revision = Revision(\n            id=self.next_id(),\n            commit_sha=commit_sha,\n            deployed_at=datetime.utcnow(),\n            manifests_hash=hash_manifests(manifests),\n            sync_status='synced'\n        )\n        self.storage.save(self.app_name, revision)\n        return revision\n```",
                    "level2": "Rollback:\n```python\nasync def rollback(self, app_name, revision_id):\n    revision = self.history.get(app_name, revision_id)\n    \n    # Get manifests from that revision\n    manifests = await self.get_manifests_at_commit(app_name, revision.commit_sha)\n    \n    # Sync to those manifests\n    result = await self.syncer.sync(app_name, manifests)\n    \n    # Record as new revision\n    new_revision = self.history.record(\n        revision.commit_sha,\n        manifests,\n        metadata={'rollback_from': self.history.current(app_name).id}\n    )\n    \n    return new_revision\n```",
                    "level3": "Auto-rollback on degraded:\n```python\nclass AutoRollbackPolicy:\n    def __init__(self, health_threshold_seconds=300):\n        self.threshold = health_threshold_seconds\n        self.degraded_since = {}\n    \n    async def check(self, app):\n        health = await self.health_checker.check_app(app)\n        \n        if health == HealthStatus.DEGRADED:\n            if app.name not in self.degraded_since:\n                self.degraded_since[app.name] = time.time()\n            elif time.time() - self.degraded_since[app.name] > self.threshold:\n                # Auto-rollback\n                previous = self.history.previous(app.name)\n                if previous:\n                    await self.rollback(app.name, previous.id)\n                    await self.notify(f'Auto-rolled back {app.name}')\n        else:\n            self.degraded_since.pop(app.name, None)\n```"
                }
            }
        ]
    },
    "serverless-runtime": {
        "name": "Serverless Function Runtime",
        "description": "Build a serverless function runtime with cold start optimization, auto-scaling, and isolation",
        "category": "Cloud Native",
        "difficulty": "expert",
        "estimated_hours": 80,
        "skills": [
            "Container isolation",
            "Cold start optimization",
            "Auto-scaling",
            "Request routing",
            "Resource limits",
            "Function lifecycle"
        ],
        "prerequisites": ["Containers", "HTTP servers", "Process management"],
        "learning_outcomes": [
            "Understand serverless architecture",
            "Implement function isolation and sandboxing",
            "Build efficient cold start mechanisms",
            "Design auto-scaling systems"
        ],
        "milestones": [
            {
                "name": "Function Packaging",
                "description": "Package and store function code",
                "skills": ["Code packaging", "Dependency resolution", "Storage"],
                "deliverables": [
                    "Function definition format",
                    "Code upload API",
                    "Dependency bundling",
                    "Runtime selection",
                    "Version management",
                    "Code storage (S3/local)"
                ],
                "hints": {
                    "level1": "Function definition:\n```python\n@dataclass\nclass FunctionDefinition:\n    name: str\n    runtime: str  # python3.9, nodejs18, etc.\n    handler: str  # module.function\n    memory_mb: int = 128\n    timeout_seconds: int = 30\n    environment: dict = None\n    code_uri: str = None  # s3://bucket/code.zip\n\nclass FunctionRegistry:\n    def create(self, definition: FunctionDefinition, code_bytes: bytes):\n        # Store code\n        code_uri = self.storage.upload(f'functions/{definition.name}/{uuid4()}.zip', code_bytes)\n        definition.code_uri = code_uri\n        \n        # Save definition\n        self.db.functions.insert(asdict(definition))\n        return definition\n```",
                    "level2": "Build function container image:\n```python\nclass FunctionBuilder:\n    RUNTIME_IMAGES = {\n        'python3.9': 'python:3.9-slim',\n        'python3.11': 'python:3.11-slim',\n        'nodejs18': 'node:18-slim',\n    }\n    \n    def build(self, function):\n        dockerfile = f'''\n            FROM {self.RUNTIME_IMAGES[function.runtime]}\n            COPY bootstrap /bootstrap\n            COPY code /var/task\n            WORKDIR /var/task\n            RUN pip install -r requirements.txt || true\n            CMD [\"/bootstrap\"]\n        '''\n        \n        # Build image\n        image_tag = f'functions/{function.name}:{function.version}'\n        self.docker.build(dockerfile, tag=image_tag)\n        return image_tag\n```",
                    "level3": "Runtime bootstrap:\n```python\n# bootstrap.py - Runs inside container\nimport importlib\nimport json\nimport sys\n\ndef main():\n    handler_path = os.environ['_HANDLER']  # module.function\n    module_name, func_name = handler_path.rsplit('.', 1)\n    \n    module = importlib.import_module(module_name)\n    handler = getattr(module, func_name)\n    \n    # Wait for invocations via HTTP\n    while True:\n        event = get_next_invocation()  # HTTP call to runtime API\n        try:\n            result = handler(event['body'], event['context'])\n            send_response(event['request_id'], result)\n        except Exception as e:\n            send_error(event['request_id'], str(e))\n```"
                }
            },
            {
                "name": "Execution Environment",
                "description": "Isolated function execution with resource limits",
                "skills": ["Container runtime", "Resource limits", "Isolation"],
                "deliverables": [
                    "Container-based isolation",
                    "Memory limits",
                    "CPU limits",
                    "Network isolation",
                    "Filesystem isolation",
                    "Execution timeout"
                ],
                "hints": {
                    "level1": "Create execution sandbox:\n```python\nimport docker\n\nclass Sandbox:\n    def __init__(self, function):\n        self.function = function\n        self.client = docker.from_env()\n        self.container = None\n    \n    def start(self):\n        self.container = self.client.containers.run(\n            image=self.function.image,\n            detach=True,\n            mem_limit=f'{self.function.memory_mb}m',\n            cpu_period=100000,\n            cpu_quota=int(self.function.memory_mb / 1769 * 100000),  # Proportional CPU\n            network_mode='none',  # No network by default\n            read_only=True,\n            environment=self.function.environment\n        )\n```",
                    "level2": "Execution with timeout:\n```python\nimport asyncio\n\nclass FunctionExecutor:\n    async def invoke(self, sandbox, event):\n        try:\n            result = await asyncio.wait_for(\n                self.send_invocation(sandbox, event),\n                timeout=sandbox.function.timeout_seconds\n            )\n            return InvocationResult(status='success', body=result)\n        except asyncio.TimeoutError:\n            sandbox.kill()\n            return InvocationResult(status='timeout')\n        except Exception as e:\n            return InvocationResult(status='error', error=str(e))\n    \n    async def send_invocation(self, sandbox, event):\n        # Send event to container's runtime API\n        async with aiohttp.ClientSession() as session:\n            resp = await session.post(\n                f'http://{sandbox.ip}:8080/invoke',\n                json=event\n            )\n            return await resp.json()\n```",
                    "level3": "gVisor for stronger isolation:\n```python\nclass GVisorSandbox(Sandbox):\n    def start(self):\n        # Use gVisor runsc runtime\n        self.container = self.client.containers.run(\n            image=self.function.image,\n            detach=True,\n            runtime='runsc',  # gVisor\n            mem_limit=f'{self.function.memory_mb}m',\n            security_opt=['no-new-privileges'],\n            cap_drop=['ALL'],\n        )\n```"
                }
            },
            {
                "name": "Cold Start Optimization",
                "description": "Minimize function startup latency",
                "skills": ["Warm pools", "Snapshotting", "Pre-warming"],
                "deliverables": [
                    "Warm container pool",
                    "Container reuse",
                    "Pre-initialization",
                    "Snapshot/restore (CRIU)",
                    "Predictive warming",
                    "Shared layers"
                ],
                "hints": {
                    "level1": "Warm pool:\n```python\nclass WarmPool:\n    def __init__(self, max_idle_per_function=3):\n        self.pools = defaultdict(list)  # function_name -> [Sandbox]\n        self.max_idle = max_idle_per_function\n    \n    def acquire(self, function_name):\n        pool = self.pools[function_name]\n        if pool:\n            return pool.pop()\n        return None  # Need cold start\n    \n    def release(self, sandbox):\n        pool = self.pools[sandbox.function.name]\n        if len(pool) < self.max_idle:\n            sandbox.reset()  # Clear state\n            pool.append(sandbox)\n        else:\n            sandbox.destroy()\n```",
                    "level2": "Pre-warming based on schedule:\n```python\nclass PreWarmer:\n    def __init__(self, warm_pool, metrics):\n        self.pool = warm_pool\n        self.metrics = metrics\n    \n    async def run(self):\n        while True:\n            # Analyze invocation patterns\n            patterns = self.metrics.get_invocation_patterns()\n            \n            for function_name, predicted_count in patterns.items():\n                current = len(self.pool.pools[function_name])\n                needed = min(predicted_count, self.pool.max_idle) - current\n                \n                if needed > 0:\n                    function = await self.get_function(function_name)\n                    for _ in range(needed):\n                        sandbox = await self.create_warm_sandbox(function)\n                        self.pool.release(sandbox)\n            \n            await asyncio.sleep(60)\n```",
                    "level3": "Snapshot with CRIU:\n```python\nclass SnapshotManager:\n    def create_snapshot(self, sandbox, snapshot_path):\n        # Checkpoint container state with CRIU\n        subprocess.run([\n            'docker', 'checkpoint', 'create',\n            '--checkpoint-dir', snapshot_path,\n            sandbox.container.id,\n            'init-snapshot'\n        ])\n    \n    def restore_from_snapshot(self, function, snapshot_path):\n        container = subprocess.run([\n            'docker', 'start',\n            '--checkpoint', 'init-snapshot',\n            '--checkpoint-dir', snapshot_path,\n            function.container_id\n        ])\n        return Sandbox(function, container)\n\n# Create snapshot after init, restore for each invocation\n```"
                }
            },
            {
                "name": "Request Routing",
                "description": "Route requests to function instances",
                "skills": ["Load balancing", "Request queuing", "Concurrency"],
                "deliverables": [
                    "HTTP gateway",
                    "Request queue per function",
                    "Concurrency limits",
                    "Load balancing",
                    "Request timeout",
                    "Async invocation"
                ],
                "hints": {
                    "level1": "Gateway routing:\n```python\n@app.post('/functions/{function_name}/invoke')\nasync def invoke(function_name: str, request: Request):\n    function = await get_function(function_name)\n    if not function:\n        raise HTTPException(404, 'Function not found')\n    \n    # Get or create sandbox\n    sandbox = warm_pool.acquire(function_name)\n    if not sandbox:\n        sandbox = await create_sandbox(function)\n    \n    try:\n        event = {\n            'body': await request.body(),\n            'headers': dict(request.headers),\n            'method': request.method\n        }\n        result = await executor.invoke(sandbox, event)\n        return Response(content=result.body, status_code=200)\n    finally:\n        warm_pool.release(sandbox)\n```",
                    "level2": "Concurrency control:\n```python\nclass ConcurrencyLimiter:\n    def __init__(self):\n        self.semaphores = {}  # function -> Semaphore\n        self.queues = {}  # function -> Queue\n    \n    async def acquire(self, function_name, max_concurrent):\n        if function_name not in self.semaphores:\n            self.semaphores[function_name] = asyncio.Semaphore(max_concurrent)\n        \n        await self.semaphores[function_name].acquire()\n    \n    def release(self, function_name):\n        self.semaphores[function_name].release()\n\n# Usage\nasync def invoke_with_limit(function, event):\n    await limiter.acquire(function.name, function.max_concurrency)\n    try:\n        return await executor.invoke(function, event)\n    finally:\n        limiter.release(function.name)\n```",
                    "level3": "Async invocation:\n```python\n@app.post('/functions/{function_name}/invoke-async')\nasync def invoke_async(function_name: str, request: Request):\n    invocation_id = str(uuid4())\n    \n    # Queue for async processing\n    await invocation_queue.put({\n        'id': invocation_id,\n        'function': function_name,\n        'event': await request.json(),\n        'callback_url': request.headers.get('X-Callback-URL')\n    })\n    \n    return {'invocation_id': invocation_id, 'status': 'queued'}\n\n# Background worker\nasync def process_async_invocations():\n    while True:\n        invocation = await invocation_queue.get()\n        result = await invoke(invocation['function'], invocation['event'])\n        \n        # Store result\n        await store_result(invocation['id'], result)\n        \n        # Callback if specified\n        if invocation.get('callback_url'):\n            await send_callback(invocation['callback_url'], result)\n```"
                }
            },
            {
                "name": "Auto-Scaling",
                "description": "Scale function instances based on demand",
                "skills": ["Metrics collection", "Scaling algorithms", "Scale to zero"],
                "deliverables": [
                    "Request rate metrics",
                    "Concurrent execution tracking",
                    "Scale-up triggers",
                    "Scale-down with cooldown",
                    "Scale to zero",
                    "Provisioned concurrency"
                ],
                "hints": {
                    "level1": "Metrics collection:\n```python\nclass FunctionMetrics:\n    def __init__(self):\n        self.invocations = defaultdict(list)  # function -> [timestamp]\n        self.concurrent = defaultdict(int)  # function -> count\n    \n    def record_invocation(self, function_name):\n        self.invocations[function_name].append(time.time())\n        # Keep last 5 minutes\n        cutoff = time.time() - 300\n        self.invocations[function_name] = [\n            t for t in self.invocations[function_name] if t > cutoff\n        ]\n    \n    def get_request_rate(self, function_name, window_seconds=60):\n        cutoff = time.time() - window_seconds\n        count = sum(1 for t in self.invocations[function_name] if t > cutoff)\n        return count / window_seconds\n```",
                    "level2": "Auto-scaler:\n```python\nclass AutoScaler:\n    def __init__(self, metrics, warm_pool):\n        self.metrics = metrics\n        self.pool = warm_pool\n        self.target_concurrency = 10  # requests per instance\n    \n    async def scale(self, function):\n        rate = self.metrics.get_request_rate(function.name)\n        current_instances = len(self.pool.pools[function.name])\n        \n        desired = max(1, int(rate / self.target_concurrency))\n        desired = min(desired, function.max_instances)\n        \n        if desired > current_instances:\n            # Scale up\n            for _ in range(desired - current_instances):\n                sandbox = await self.create_sandbox(function)\n                self.pool.release(sandbox)\n        elif desired < current_instances and self.can_scale_down(function):\n            # Scale down\n            excess = current_instances - desired\n            for _ in range(excess):\n                sandbox = self.pool.acquire(function.name)\n                if sandbox:\n                    sandbox.destroy()\n```",
                    "level3": "Scale to zero with cold start tracking:\n```python\nclass ScaleToZeroPolicy:\n    def __init__(self, idle_timeout=300):\n        self.idle_timeout = idle_timeout\n        self.last_invocation = {}\n    \n    def should_scale_to_zero(self, function_name):\n        last = self.last_invocation.get(function_name, 0)\n        return time.time() - last > self.idle_timeout\n    \n    async def run(self):\n        while True:\n            for function_name, instances in self.pool.pools.items():\n                if instances and self.should_scale_to_zero(function_name):\n                    # Destroy all instances\n                    while instances:\n                        sandbox = instances.pop()\n                        sandbox.destroy()\n                    logger.info(f'Scaled {function_name} to zero')\n            \n            await asyncio.sleep(60)\n```"
                }
            }
        ]
    },
    "mlops-platform": {
        "name": "MLOps Platform",
        "description": "Build an end-to-end MLOps platform for training, versioning, deploying, and monitoring ML models",
        "category": "AI/ML Infrastructure",
        "difficulty": "advanced",
        "estimated_hours": 70,
        "skills": [
            "Experiment tracking",
            "Model versioning",
            "Pipeline orchestration",
            "Model deployment",
            "Feature store",
            "Model monitoring"
        ],
        "prerequisites": ["ML basics", "Docker", "REST APIs"],
        "learning_outcomes": [
            "Design ML lifecycle management systems",
            "Implement experiment tracking and versioning",
            "Build automated training pipelines",
            "Deploy and monitor models in production"
        ],
        "milestones": [
            {
                "name": "Experiment Tracking",
                "description": "Track experiments, parameters, and metrics",
                "skills": ["Logging", "Metadata storage", "Visualization"],
                "deliverables": [
                    "Experiment and run abstractions",
                    "Parameter logging",
                    "Metric logging with steps",
                    "Artifact storage",
                    "Run comparison",
                    "Experiment UI"
                ],
                "hints": {
                    "level1": "Experiment tracking API:\n```python\nclass Experiment:\n    def __init__(self, name, tracking_uri):\n        self.name = name\n        self.tracking_uri = tracking_uri\n        self.run_id = None\n    \n    def start_run(self, run_name=None):\n        self.run_id = str(uuid4())\n        self.run = Run(\n            id=self.run_id,\n            experiment=self.name,\n            name=run_name,\n            start_time=datetime.utcnow()\n        )\n        return self\n    \n    def log_param(self, key, value):\n        self.run.params[key] = value\n    \n    def log_metric(self, key, value, step=None):\n        self.run.metrics.append(Metric(key, value, step, time.time()))\n    \n    def log_artifact(self, local_path, artifact_path=None):\n        dest = f'{self.tracking_uri}/artifacts/{self.run_id}/{artifact_path or os.path.basename(local_path)}'\n        shutil.copy(local_path, dest)\n```",
                    "level2": "Auto-logging decorator:\n```python\ndef autolog(framework='sklearn'):\n    def decorator(train_func):\n        def wrapper(*args, **kwargs):\n            with mlops.start_run():\n                # Log function params\n                mlops.log_params(kwargs)\n                \n                result = train_func(*args, **kwargs)\n                \n                # Framework-specific logging\n                if framework == 'sklearn' and hasattr(result, 'get_params'):\n                    mlops.log_params(result.get_params())\n                \n                return result\n        return wrapper\n    return decorator\n\n@autolog('sklearn')\ndef train_model(X, y, n_estimators=100):\n    model = RandomForestClassifier(n_estimators=n_estimators)\n    model.fit(X, y)\n    return model\n```",
                    "level3": "Run comparison:\n```python\nclass ExperimentAnalyzer:\n    def compare_runs(self, run_ids):\n        runs = [self.get_run(rid) for rid in run_ids]\n        \n        # Parameter diff\n        all_params = set()\n        for run in runs:\n            all_params.update(run.params.keys())\n        \n        param_diff = {}\n        for param in all_params:\n            values = [run.params.get(param) for run in runs]\n            if len(set(values)) > 1:\n                param_diff[param] = values\n        \n        # Metric comparison\n        metric_comparison = {}\n        for run in runs:\n            for metric in run.metrics:\n                if metric.key not in metric_comparison:\n                    metric_comparison[metric.key] = []\n                metric_comparison[metric.key].append({\n                    'run_id': run.id,\n                    'value': metric.value\n                })\n        \n        return {'param_diff': param_diff, 'metrics': metric_comparison}\n```"
                }
            },
            {
                "name": "Model Registry",
                "description": "Version and manage trained models",
                "skills": ["Model serialization", "Versioning", "Stage management"],
                "deliverables": [
                    "Model registration",
                    "Version management",
                    "Stage transitions (staging/production)",
                    "Model metadata",
                    "Model lineage",
                    "Model download API"
                ],
                "hints": {
                    "level1": "Model registry:\n```python\n@dataclass\nclass RegisteredModel:\n    name: str\n    versions: List['ModelVersion']\n    description: str = None\n    tags: dict = None\n\n@dataclass\nclass ModelVersion:\n    version: int\n    model_uri: str\n    run_id: str\n    stage: str = 'None'  # None, Staging, Production, Archived\n    created_at: datetime = None\n\nclass ModelRegistry:\n    def register_model(self, name, model_uri, run_id=None):\n        model = self.get_or_create_model(name)\n        version = ModelVersion(\n            version=len(model.versions) + 1,\n            model_uri=model_uri,\n            run_id=run_id,\n            created_at=datetime.utcnow()\n        )\n        model.versions.append(version)\n        self.save(model)\n        return version\n```",
                    "level2": "Stage transitions:\n```python\ndef transition_stage(self, name, version, stage):\n    model = self.get_model(name)\n    version_obj = model.versions[version - 1]\n    \n    # If transitioning to Production, archive current production\n    if stage == 'Production':\n        for v in model.versions:\n            if v.stage == 'Production':\n                v.stage = 'Archived'\n    \n    version_obj.stage = stage\n    self.save(model)\n    \n    # Trigger webhooks\n    self.notify_stage_change(name, version, stage)\n    \n    return version_obj\n```",
                    "level3": "Model lineage:\n```python\nclass ModelLineage:\n    def get_lineage(self, model_name, version):\n        model_version = self.registry.get_version(model_name, version)\n        run = self.tracking.get_run(model_version.run_id)\n        \n        return {\n            'model': {'name': model_name, 'version': version},\n            'training_run': {\n                'id': run.id,\n                'experiment': run.experiment,\n                'params': run.params,\n                'metrics': run.final_metrics()\n            },\n            'data': {\n                'dataset': run.params.get('dataset'),\n                'dataset_version': run.params.get('dataset_version')\n            },\n            'code': {\n                'git_commit': run.tags.get('git_commit'),\n                'git_repo': run.tags.get('git_repo')\n            }\n        }\n```"
                }
            },
            {
                "name": "Training Pipeline",
                "description": "Orchestrate training workflows",
                "skills": ["DAG orchestration", "Containerized training", "Distributed training"],
                "deliverables": [
                    "Pipeline definition DSL",
                    "Step execution",
                    "Data passing between steps",
                    "Conditional execution",
                    "Parallel steps",
                    "Pipeline versioning"
                ],
                "hints": {
                    "level1": "Pipeline definition:\n```python\nclass Pipeline:\n    def __init__(self, name):\n        self.name = name\n        self.steps = []\n    \n    def add_step(self, step):\n        self.steps.append(step)\n        return self\n\n@dataclass\nclass Step:\n    name: str\n    command: str\n    image: str = 'python:3.9'\n    inputs: List[str] = None\n    outputs: List[str] = None\n    resources: dict = None\n\n# Usage\npipeline = Pipeline('training')\npipeline.add_step(Step(\n    name='preprocess',\n    command='python preprocess.py --input {input_data} --output {processed_data}',\n    outputs=['processed_data']\n))\npipeline.add_step(Step(\n    name='train',\n    command='python train.py --data {processed_data} --model {model_output}',\n    inputs=['processed_data'],\n    outputs=['model_output'],\n    resources={'gpu': 1}\n))\n```",
                    "level2": "Pipeline executor:\n```python\nclass PipelineExecutor:\n    async def run(self, pipeline, params):\n        context = {'params': params, 'outputs': {}}\n        \n        for step in self.topological_sort(pipeline.steps):\n            # Wait for dependencies\n            for input_name in step.inputs or []:\n                if input_name not in context['outputs']:\n                    raise ValueError(f'Missing input: {input_name}')\n            \n            # Execute step\n            result = await self.run_step(step, context)\n            context['outputs'].update(result)\n        \n        return context['outputs']\n    \n    async def run_step(self, step, context):\n        # Render command with context\n        command = step.command.format(**context['params'], **context['outputs'])\n        \n        # Run in container\n        container = await self.docker.run(\n            image=step.image,\n            command=command,\n            resources=step.resources\n        )\n        \n        return {out: container.get_output(out) for out in step.outputs or []}\n```",
                    "level3": "Pipeline versioning and triggers:\n```python\nclass PipelineRegistry:\n    def register(self, pipeline):\n        version = self.next_version(pipeline.name)\n        pipeline_def = {\n            'name': pipeline.name,\n            'version': version,\n            'steps': [asdict(s) for s in pipeline.steps],\n            'created_at': datetime.utcnow()\n        }\n        self.store.save(pipeline.name, version, pipeline_def)\n        return version\n    \n    def trigger_on_data_change(self, pipeline_name, data_path):\n        # Register webhook for data changes\n        self.webhooks.register(\n            event='data.updated',\n            filter={'path': data_path},\n            action={'pipeline': pipeline_name, 'trigger': 'run'}\n        )\n```"
                }
            },
            {
                "name": "Model Deployment",
                "description": "Deploy models to production",
                "skills": ["Serving infrastructure", "Canary deployment", "A/B testing"],
                "deliverables": [
                    "Model serving endpoint",
                    "Blue-green deployment",
                    "Canary rollout",
                    "A/B testing",
                    "Auto-scaling",
                    "Deployment rollback"
                ],
                "hints": {
                    "level1": "Deploy model:\n```python\nclass ModelDeployer:\n    def deploy(self, model_name, version, endpoint_name):\n        model_version = self.registry.get_version(model_name, version)\n        \n        # Create serving container\n        deployment = Deployment(\n            name=endpoint_name,\n            image='model-server:latest',\n            env={\n                'MODEL_URI': model_version.model_uri,\n                'MODEL_NAME': model_name\n            },\n            replicas=2,\n            resources={'cpu': '500m', 'memory': '1Gi'}\n        )\n        \n        self.k8s.create_deployment(deployment)\n        self.k8s.create_service(endpoint_name)\n        \n        return Endpoint(name=endpoint_name, url=f'http://{endpoint_name}/predict')\n```",
                    "level2": "Canary deployment:\n```python\ndef deploy_canary(self, model_name, new_version, endpoint_name, canary_percent=10):\n    current = self.get_deployment(endpoint_name)\n    \n    # Create canary deployment\n    canary = self.deploy(\n        model_name, new_version,\n        endpoint_name=f'{endpoint_name}-canary'\n    )\n    \n    # Update traffic split\n    self.update_traffic_split(endpoint_name, {\n        current.name: 100 - canary_percent,\n        canary.name: canary_percent\n    })\n    \n    return CanaryDeployment(primary=current, canary=canary, percent=canary_percent)\n\ndef promote_canary(self, endpoint_name):\n    # Shift all traffic to canary\n    self.update_traffic_split(endpoint_name, {\n        f'{endpoint_name}-canary': 100\n    })\n    # Rename canary to primary\n    self.rename_deployment(f'{endpoint_name}-canary', endpoint_name)\n```",
                    "level3": "A/B test with metrics:\n```python\nclass ABTest:\n    def __init__(self, name, variants, metric):\n        self.name = name\n        self.variants = variants  # {name: model_version}\n        self.metric = metric\n        self.results = defaultdict(list)\n    \n    def assign_variant(self, user_id):\n        # Consistent hashing for user\n        hash_val = hash(f'{self.name}:{user_id}') % 100\n        cumulative = 0\n        for variant, percent in self.variants.items():\n            cumulative += percent\n            if hash_val < cumulative:\n                return variant\n    \n    def record_outcome(self, user_id, outcome):\n        variant = self.assign_variant(user_id)\n        self.results[variant].append(outcome)\n    \n    def get_results(self):\n        from scipy import stats\n        control = self.results['control']\n        treatment = self.results['treatment']\n        \n        t_stat, p_value = stats.ttest_ind(control, treatment)\n        return {\n            'control_mean': np.mean(control),\n            'treatment_mean': np.mean(treatment),\n            'p_value': p_value,\n            'significant': p_value < 0.05\n        }\n```"
                }
            },
            {
                "name": "Model Monitoring",
                "description": "Monitor model performance in production",
                "skills": ["Metrics collection", "Drift detection", "Alerting"],
                "deliverables": [
                    "Prediction logging",
                    "Performance metrics",
                    "Data drift detection",
                    "Model drift detection",
                    "Alerting rules",
                    "Monitoring dashboard"
                ],
                "hints": {
                    "level1": "Prediction logging:\n```python\nclass PredictionLogger:\n    def __init__(self, kafka_producer):\n        self.producer = kafka_producer\n    \n    def log(self, model_name, version, input_data, prediction, latency):\n        record = {\n            'timestamp': datetime.utcnow().isoformat(),\n            'model': model_name,\n            'version': version,\n            'input_hash': hash(str(input_data)),\n            'input_sample': self.sample_input(input_data),\n            'prediction': prediction,\n            'latency_ms': latency\n        }\n        self.producer.send('predictions', record)\n    \n    def log_feedback(self, prediction_id, ground_truth):\n        self.producer.send('feedback', {\n            'prediction_id': prediction_id,\n            'ground_truth': ground_truth\n        })\n```",
                    "level2": "Data drift detection:\n```python\nclass DriftMonitor:\n    def __init__(self, reference_data):\n        self.reference_stats = self.compute_stats(reference_data)\n    \n    def check_drift(self, current_data):\n        current_stats = self.compute_stats(current_data)\n        drift_scores = {}\n        \n        for feature in self.reference_stats:\n            # PSI (Population Stability Index)\n            ref = self.reference_stats[feature]['distribution']\n            curr = current_stats[feature]['distribution']\n            psi = self.compute_psi(ref, curr)\n            drift_scores[feature] = psi\n        \n        return {\n            'drift_scores': drift_scores,\n            'has_drift': any(s > 0.2 for s in drift_scores.values())\n        }\n    \n    def compute_psi(self, expected, actual):\n        psi = 0\n        for i in range(len(expected)):\n            if expected[i] == 0 or actual[i] == 0:\n                continue\n            psi += (actual[i] - expected[i]) * np.log(actual[i] / expected[i])\n        return psi\n```",
                    "level3": "Alerting:\n```python\nclass AlertManager:\n    def __init__(self):\n        self.rules = []\n    \n    def add_rule(self, name, condition, action):\n        self.rules.append(AlertRule(name, condition, action))\n    \n    async def evaluate(self, metrics):\n        for rule in self.rules:\n            if rule.condition(metrics):\n                await rule.action.execute({\n                    'rule': rule.name,\n                    'metrics': metrics\n                })\n\n# Example rules\nalert_manager.add_rule(\n    'high_latency',\n    condition=lambda m: m['p99_latency'] > 500,\n    action=SlackAlert(channel='#ml-alerts')\n)\nalert_manager.add_rule(\n    'accuracy_drop',\n    condition=lambda m: m['accuracy'] < 0.9,\n    action=PagerDutyAlert(severity='high')\n)\nalert_manager.add_rule(\n    'data_drift',\n    condition=lambda m: m['drift_score'] > 0.2,\n    action=RetrainTrigger(pipeline='training')\n)\n```"
                }
            }
        ]
    }
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    if 'expert_projects' not in data:
        data['expert_projects'] = {}

    added = 0
    for project_id, project_data in DETAILED_PROJECTS.items():
        data['expert_projects'][project_id] = project_data
        print(f"Added detailed milestones: {project_id}")
        added += 1

    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"\nAdded detailed milestones for {added} more projects")


if __name__ == "__main__":
    main()
