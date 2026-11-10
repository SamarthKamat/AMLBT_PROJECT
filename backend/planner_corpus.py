"""Training corpus for the local intent classifier (Phase 2).

Questions are written over MASKED schema tokens, exactly as the local planner sees them at run time after schema
linking: numcol (a numeric column), catcol (a categorical column), datecol, valtok (a category value), numtok
(a number). So the classifier learns the STRUCTURE of a request, not particular column names, and generalises to
datasets it has never seen. "{a|b|c}" marks alternatives; every template is expanded by random sampling.

The held-out evaluation questions (evaluation/benchmark_questions.jsonl) are written separately in natural language
over real column names; none of them is generated from these templates.
"""
from __future__ import annotations
import itertools
import random
import re
from typing import Dict, List, Tuple

T: Dict[str, List[str]] = {
    "get_summary": [
        "what is the {average|mean|avg|typical} numcol", "{average|mean} numcol", "what is the {total|sum of|overall} numcol", "{total|combined} numcol",
        "how many {rows|records|entries|orders|employees|customers|transactions} are there", "how many {rows|records} {does the data have|are in the dataset}",
        "count of {rows|records}", "what is the {median|middle} numcol", "what is the {maximum|highest|largest|max} numcol", "what is the {minimum|lowest|smallest|min} numcol",
        "{maximum|minimum} numcol {in the data|recorded|overall}", "what is the standard deviation of numcol", "how {variable|spread out} is numcol",
        "{summarise|summarize|describe} numcol", "give me {a summary|summary statistics|basic stats} {of|for} numcol", "{stats|statistics} for numcol",
        "what is the distribution of catcol", "{breakdown|split|composition|mix} of catcol", "what share of {records|rows} {are|have} {each|every} catcol",
        "what percentage of {records|rows} {is|are} in each catcol", "how many {unique|distinct|different} catcol {are there|exist}", "number of {unique|distinct} catcol",
        "how is catcol distributed", "distribution of catcol", "catcol distribution", "{show|give} the distribution of catcol", "{breakdown|split} of {records|rows} by catcol",
        "what is the {share|percentage} of each catcol", "{plot|chart} the distribution of catcol", "{give me|show} an overview of {the|this} {data|dataset}", "{summarise|summarize} the {data|dataset}",
        "what does the dataset contain", "what is the numcol {on average|per record}", "how much numcol {is there|in total}", "sum of numcol",
        "what is the {average|total} numcol {for|in|of} valtok", "{average|total} numcol {for|in} valtok", "how many {records|rows|orders} {are|have} valtok",
        "what proportion {are|is} valtok", "what percentage of catcol is valtok", "how many valtok {are there|records}",
    ],
    "get_current_state": [
        "what is the {latest|current|most recent} numcol", "{latest|current|most recent} value of numcol", "what was numcol {on the last day|most recently|last recorded}",
        "show the {latest|most recent} {record|records|entry|observation}", "what is numcol {right now|currently|today}", "{latest|current} numcol for each catcol",
        "what is the most recent numcol by catcol", "what is the {latest|current} status", "when was the {last|latest} {record|entry}",
        "{current|latest} numcol {for|in} valtok", "what {is|was} the last reported numcol",
    ],
    "compare_groups": [
        "which catcol has the {highest|most|largest|biggest|greatest|maximum} numcol", "which catcol has the {lowest|least|smallest|fewest|minimum} numcol",
        "{compare|comparison of} numcol {across|between|by|for each|among} catcol", "numcol by catcol", "{total|average|mean|median} numcol {by|per|for each|across} catcol",
        "{rank|ranking of} catcol by numcol", "{top|best} numtok catcol by numcol", "{bottom|worst} numtok catcol {by|for} numcol",
        "which catcol {performs|does} {best|worst} {on|in terms of} numcol", "which catcol {sells|earns|generates|makes} the most", "which catcol has the most {records|rows|orders|entries}",
        "how does numcol {vary|differ|change} {across|by|between} catcol", "is numcol {different|higher|lower} {for|in} some catcol", "{show|plot|chart} numcol {by|per} catcol",
        "numcol {per|by|for each} catcol and catcol", "{compare|contrast} catcol {on|by} numcol", "{where|in which catcol} is numcol {highest|lowest}",
        "{highest|lowest} numcol catcol", "{which|what} catcol {has|had|shows} the {best|worst} numcol", "{leading|top performing|worst performing} catcol {for|by} numcol",
        "how many {records|rows|orders|employees|customers} {per|by|in each} catcol", "count {of records|of rows} by catcol", "which catcol is the {largest|biggest|most common}",
        "numcol {split|broken down} by catcol", "{average|total} numcol for each catcol {in|for} valtok", "compare catcol {for|in} valtok by numcol",
        "what is the {share|proportion|percentage} of catcol {in|for} each catcol", "catcol mix by catcol", "how does catcol {vary|differ} by catcol",
        "{rate|percentage} of valtok {by|per|for each|across} catcol", "which catcol has the highest {rate|share} of valtok",
    ],
    "analyze_trend": [
        "is numcol {increasing|decreasing|rising|falling|growing|declining|going up|going down}", "{trend|trends} {of|in} numcol", "how has numcol {changed|evolved|moved|developed} over time",
        "numcol over time", "{show|plot|chart} numcol over {time|the months|the years|the weeks}", "{monthly|weekly|daily|quarterly|yearly} numcol", "numcol {per|by|each} {month|week|day|quarter|year}",
        "is there a trend in numcol", "{has|did} numcol {grown|declined|improved|worsened} over time", "what is the trend of numcol {by|for each} catcol",
        "numcol {trend|over time} for valtok", "how did numcol {develop|change} {month by month|year over year|week by week}", "is numcol getting {better|worse}",
        "{growth|decline} of numcol over time", "time series of numcol", "how many {records|orders} {per|each} {month|week|day} over time", "evolution of numcol",
    ],
    "detect_anomalies": [
        "{are there|find|show|detect|identify|list} {any|} {outliers|anomalies|unusual values|abnormal values|extreme values|odd values|strange values} in numcol",
        "which {records|rows|orders|transactions|employees|customers} {look|are|seem} {unusual|abnormal|suspicious|odd|anomalous}", "{outliers|anomalies} {in|of} numcol",
        "{detect|find|spot} {anomalies|outliers}", "{any|are there} {suspicious|fraudulent|weird} {transactions|records|entries}", "is any numcol {too high|too low|unusual}",
        "{flag|highlight} unusual numcol", "{unusually|abnormally} {high|low} numcol", "extreme numcol values", "anomaly detection on {the data|numcol}",
        "which {records|rows} are outliers", "{show|find} records with abnormal numcol",
    ],
    "calculate_correlation": [
        "what is {related|correlated|associated|linked} {to|with} numcol", "{correlation|relationship|association} between numcol and numcol", "how does numcol relate to numcol",
        "is numcol {correlated|related|associated} with numcol", "what {factors|variables|columns} {affect|influence|drive|are associated with|impact} numcol", "{what drives|drivers of} numcol",
        "does numcol {depend on|go up with|move with} numcol", "{correlation|correlations} {matrix|of numeric columns|in the data}", "which variables are most correlated",
        "is there a {link|connection|relationship} between catcol and numcol", "how strongly is numcol tied to numcol", "what {affects|influences|explains} catcol",
    ],
    "describe_relationships": [
        "how are the columns {related|connected|linked}", "{show|explain|describe} the {relationships|connections|links} between {the columns|columns|variables|fields}",
        "{column|columns} {relationships|relationship map|dependencies}", "{data dictionary|dictionary of the data|schema}", "what does each column {mean|represent|contain}",
        "explain the {columns|fields|variables} {in the dataset|of the data|}", "which columns are {linked|dependent|redundant|derived}", "how does catcol relate to {the other columns|other columns}",
        "{relationships|relations} {of|for} catcol", "{relationships|relations} {of|for} numcol {with the other columns|}", "are there {hierarchies|dependencies|derived columns} {in the data|between columns}",
        "column wise {relations|relationships}", "map of how {variables|columns} relate", "{describe|explain} the {structure|schema} of the {data|dataset}",
    ],
    "predict_value": [
        "predict numcol", "{estimate|predict|calculate} {the|} numcol for {a|an} {record|order|employee|customer} with numcol numtok", "what would numcol be {if|when|for} numcol is numtok",
        "{build|train|create} a {model|regression} {to predict|for|that predicts} numcol", "{predicted|expected|estimated} numcol {for|when} catcol is valtok",
        "how much numcol {would|will|should} {a|an} valtok {record|employee|customer|order} {have|get|make}", "predict numcol {given|from|using} {the other columns|catcol and numcol}",
        "estimate numcol for valtok with numcol numtok", "what numcol {can we expect|should we expect} for numcol numtok", "regression model for numcol",
        "can you predict numcol", "{model|forecast} numcol from {the other variables|other columns}", "what is the expected numcol for valtok",
    ],
    "predict_class": [
        "{will|would} {a|an|this} {employee|customer|record|order|client} {churn|leave|default|be late|convert|quit|subscribe}", "predict catcol", "{predict|classify} whether catcol",
        "{which|what} {employees|customers|records|orders} are likely to {churn|leave|default|quit|be late|convert}", "{probability|likelihood|chance} of {churn|attrition|default|catcol|valtok}",
        "{build|train} a {classifier|classification model} for catcol", "is {a|an} valtok {employee|customer|record} likely to {churn|leave|be valtok}", "predict {if|whether} catcol is valtok",
        "{who|which customers|which employees} {will|might} {leave|churn|quit}", "{classify|predict the class of} catcol", "what predicts catcol", "what predicts {churn|attrition|default}",
        "how likely is valtok {for|given} catcol valtok", "predict catcol for valtok", "risk of catcol",
    ],
    "forecast": [
        "forecast numcol {for the next|over the next} numtok {days|weeks|months|quarters|years}", "{forecast|project|predict} numcol for next {month|week|quarter|year}",
        "what will numcol be {next month|next week|next quarter|next year|in the coming months}", "{projection|projections|forecast} of numcol", "{forecast|predict} future numcol",
        "how much numcol {will we have|can we expect} next {month|quarter|year}", "{expected|projected} numcol {next|for the next} numtok {months|weeks}", "forecast {demand|sales|numcol}",
        "numcol forecast", "{predict|estimate} numcol for the {coming|upcoming|following} {weeks|months|days}", "{forecast|project} {monthly|weekly|daily} numcol",
        "future {trend|values} of numcol", "what is the outlook for numcol",
    ],
    "run_scenario": [
        "what if numcol {increases|rises|grows|goes up} by numtok percent", "what if numcol {decreases|drops|falls|goes down} by numtok percent", "what happens to numcol if numcol {increases|decreases|doubles|falls} by numtok percent",
        "{simulate|model} {a|} numtok percent {increase|decrease|rise|cut|drop} in numcol", "if we {raise|cut|increase|reduce|lower} numcol by numtok percent what happens to numcol",
        "impact {on|of} numcol {of|if} numcol {changes|increases|drops} by numtok percent", "{scenario|what if analysis|sensitivity analysis} {for|on} numcol",
        "how would numcol change if numcol were numtok", "what if numcol {was|were|is} numtok", "effect of {increasing|reducing|raising|cutting} numcol by numtok on numcol",
        "suppose numcol {goes up|goes down} numtok percent", "{increase|decrease|raise|reduce} numcol by numtok percent and show numcol",
    ],
    "query_records": [
        "{show|list|display|give me|find} {the|all|} {records|rows|orders|employees|customers|transactions|entries} {where|with|having|in|for} valtok",
        "{show|list|display} {records|rows|orders} {where|with} numcol {above|over|greater than|more than|below|under|less than} numtok",
        "{top|first} numtok {records|rows|orders|transactions|customers|employees} by numcol", "{show|list} the numtok {largest|biggest|highest|smallest|lowest} {orders|records|rows|transactions} by numcol",
        "which {orders|records|rows|transactions|employees|customers} have numcol {above|over|below|under} numtok", "{find|show|list} {employees|customers|orders|records} {from|in|with} valtok and valtok",
        "{show|give} me {the|} {rows|records} for valtok", "{list|show} {records|rows} with the {highest|lowest} numcol", "{filter|select} {the data|rows|records} {where|with|for} valtok",
        "{show|list} {orders|records|rows} {from|in} valtok sorted by numcol", "{display|show} {the|} raw data {for|of} valtok", "which {employees|customers} {are|work} in valtok",
        "{show|list} {the|} {records|rows} {with|where} numcol {between|from} numtok and numtok", "{most|least} {expensive|profitable|valuable} {orders|records|transactions}",
    ],
    "pivot_table": [
        "{pivot|pivot table|cross tab|crosstab|matrix|cross tabulation} of numcol by catcol and catcol", "numcol by catcol and catcol", "{total|average} numcol {by|per} catcol and catcol",
        "{make|create|build|show} a pivot table {of|for|with} numcol {by|across} catcol and catcol", "{cross tab|crosstab|cross tabulate} catcol {against|by|and} catcol",
        "numcol {broken down|split} by catcol and {by|} catcol", "{heatmap|matrix} of numcol {across|by} catcol and catcol", "how many {records|rows} {by|for each} catcol and catcol",
        "{breakdown|split} of numcol by catcol {versus|vs|and} catcol",
    ],
    "run_eda": [
        "{do|perform|run|carry out|conduct} {an|} {eda|exploratory data analysis|exploratory analysis}", "{explore|analyse|analyze|investigate} {the|this} {data|dataset} {in detail|thoroughly|}",
        "{eda|exploratory analysis} {of|on|for} {the data|numcol|catcol}", "{show|plot} the distributions of {all|the} {columns|variables}", "{explore|investigate|deep dive into|analyse} numcol",
        "{explore|deep dive into|analyse} catcol", "give me {an|a} {exploratory|complete|full|detailed} {analysis|overview|picture} of the data", "what {insights|patterns} {are|can you find} in the data",
        "{univariate|bivariate} analysis", "{distribution|distributions} {and|,} {outliers|correlations} {of|in} the data", "{explore|analyse} {the shape of|} numcol in depth",
        "tell me {everything|about} {the data|this dataset}", "{analyse|analyze} the dataset", "what are the key insights",
    ],
    "assess_data_quality": [
        "{check|assess|evaluate|audit|review} {the|} data quality", "{how good|how clean|how reliable} is {the|this} {data|dataset}", "{are there|any|find} {missing values|duplicates|duplicate rows|data issues|errors|inconsistencies}",
        "data quality {score|report|check|assessment}", "{is|are} {the|} data {clean|complete|consistent|valid}", "what {problems|issues|errors} {does|are in} the {data|dataset} {have|}",
        "how many missing values {are there|in the data}", "which columns have missing values", "{should|do} i clean the data", "{what|which} cleaning is needed",
        "{find|check} {invalid|wrong|inconsistent} values", "data {issues|problems|errors}",
    ],
    "explain_change": [
        "why did numcol {drop|fall|decline|decrease|increase|rise|go up|go down|change}", "what {drove|caused|explains|is behind} the {drop|fall|decline|increase|rise|change|growth} in numcol",
        "{root cause|root-cause analysis|driver analysis} {of|for} {the|} {change|drop|decline|increase} in numcol", "explain the {change|drop|increase|decline} in numcol {last month|last quarter|recently|this year|}",
        "{where|which catcol} did the {drop|decline|increase|growth} in numcol come from", "what changed in numcol {between|from} numtok {and|to} numtok", "{why|how come} numcol {went down|went up|fell|rose} {last month|last quarter|recently|}",
        "{decompose|break down} the change in numcol", "what contributed {most|} to the {change|decline|growth} {of|in} numcol", "{diagnose|investigate} the {drop|fall|decline} in numcol",
        "why {is|are} numcol {down|up|lower|higher} {this month|this quarter|this year|than before}",
    ],
    "segment_records": [
        "{segment|cluster|group} {the|my|} {customers|employees|records|data|rows|clients}", "{customer|employee|market} segmentation", "{find|identify|discover} {segments|clusters|personas|groups of similar records}",
        "{cluster|segment} {records|customers} {by|using|on} numcol and numcol", "{split|divide} {customers|employees|records} into numtok {segments|clusters|groups}", "{run|do|perform} {clustering|k means|kmeans|segmentation}",
        "what {types|kinds} of {customers|employees} {are there|do we have}", "{group|cluster} similar {customers|records|employees} together", "numtok {segments|clusters} of {customers|employees}",
    ],
    "generate_report": [
        "{generate|create|make|prepare|produce|write|build} {a|an|the} {marketing|sales|hr|human resources|operations|finance|financial|customer|business|management|executive} report",
        "{marketing|sales|hr|operations|finance|customer|business} report", "{give|prepare} me {a|an} {executive summary|management summary|business summary|board report}",
        "report {for|on} {marketing|sales|hr|operations|finance|the business|management}", "{create|generate} {a|} {full|complete|detailed|} report {of|on|for} {the data|this dataset}",
        "{summarise|summarize} the data {for|into a report for} {the marketing team|the hr team|operations|finance|management|sales}", "{business|decision} {insights|recommendations} report",
        "what should {marketing|hr|operations|finance|sales|management} {do|focus on}", "{download|export} a report", "{report|dashboard} {with|of} {kpis|recommendations}",
    ],
    "unsupported": [
        "what is the weather {today|tomorrow|in goa}", "tell me a joke", "{send|write} an email to {the team|my manager}", "book a meeting {tomorrow|next week}", "who won the {match|election|world cup}",
        "{delete|drop|wipe} the {database|dataset|table}", "write python code {to|for} {sort a list|scrape a website}", "translate this {to|into} {french|hindi}", "what is the capital of {france|india}",
        "{play|recommend} {some|a} {music|song|movie}", "{hello|hi|hey} how are you", "what time is it", "{order|buy} {pizza|a laptop} for me", "who are you",
        "{explain|what is} {quantum physics|machine learning|blockchain} in general", "open {youtube|google}", "{what is|tell me} the stock price of {apple|tesla}",
    ],
}

_ALT = re.compile(r"\{([^{}]*)\}")


def expand(template: str, rng: random.Random, k: int) -> List[str]:
    parts = _ALT.split(template)
    slots = [parts[i].split("|") for i in range(1, len(parts), 2)]
    fixed = [parts[i] for i in range(0, len(parts), 2)]
    total = 1
    for s in slots:
        total *= len(s)
    combos = list(itertools.product(*slots)) if total <= 4 * k else [tuple(rng.choice(s) for s in slots) for _ in range(k)]
    rng.shuffle(combos)
    out = []
    for combo in combos[:k]:
        text = "".join(f + (combo[i] if i < len(combo) else "") for i, f in enumerate(fixed))
        out.append(re.sub(r"\s+", " ", text).strip())
    return out


PREFIXES = ["", "", "", "please ", "can you ", "could you ", "i want to know ", "tell me ", "show me ", "hey, ", "quick question: "]
SUFFIXES = ["", "", "", "?", " please", " in the data", " for me", " ?", " in this dataset"]


def build_corpus(per_template: int = 14, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    X, y = [], []
    for label, temps in T.items():
        for t in temps:
            for q in expand(t, rng, per_template):
                q2 = (rng.choice(PREFIXES) + q + rng.choice(SUFFIXES)).strip()
                X.append(q2.lower())
                y.append(label)
    return X, y
