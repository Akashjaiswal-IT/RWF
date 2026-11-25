"""Exact presets and paper tables from Li, Wang, & Böhm (IEEE ICDM 2024)."""

PRESET_GRAPHS = {
    "worked": {
        "name": "Paper Worked Example (Part 4, Graph 1)",
        "nodes": [
            {"id": "A", "x": 255, "y": 195, "feat": "C"},
            {"id": "B", "x": 145, "y": 110, "feat": "C"},
            {"id": "C", "x": 375, "y": 110, "feat": "N"},
            {"id": "D", "x": 120, "y": 255, "feat": "C"},
            {"id": "E", "x": 315, "y": 300, "feat": "O"},
            {"id": "F", "x": 500, "y": 85, "feat": "O"},
        ],
        "edges": [
            ["A", "B"], ["A", "C"], ["A", "D"], ["A", "E"],
            ["B", "C"], ["B", "D"], ["C", "F"],
        ],
    },
    "star": {
        "name": "Star + Edge Comparison (Part 4, Graph 2)",
        "nodes": [
            {"id": "A", "x": 300, "y": 180, "feat": "C"},
            {"id": "B", "x": 220, "y": 80, "feat": "C"},
            {"id": "C", "x": 380, "y": 80, "feat": "N"},
            {"id": "D", "x": 160, "y": 215, "feat": "O"},
            {"id": "E", "x": 300, "y": 305, "feat": "O"},
            {"id": "F", "x": 440, "y": 215, "feat": "O"},
        ],
        "edges": [
            ["A", "B"], ["A", "C"], ["A", "D"],
            ["A", "E"], ["A", "F"], ["B", "C"],
        ],
    },
    "molecule": {
        "name": "Nitro-Molecule (MUTAG-style)",
        "nodes": [
            {"id": "C1", "x": 210, "y": 140, "feat": "C"},
            {"id": "C2", "x": 290, "y": 100, "feat": "C"},
            {"id": "C3", "x": 370, "y": 140, "feat": "C"},
            {"id": "C4", "x": 370, "y": 225, "feat": "C"},
            {"id": "C5", "x": 290, "y": 265, "feat": "C"},
            {"id": "C6", "x": 210, "y": 225, "feat": "C"},
            {"id": "N1", "x": 455, "y": 182, "feat": "N"},
            {"id": "O1", "x": 525, "y": 125, "feat": "O"},
            {"id": "O2", "x": 525, "y": 240, "feat": "O"},
        ],
        "edges": [
            ["C1", "C2"], ["C2", "C3"], ["C3", "C4"], ["C4", "C5"],
            ["C5", "C6"], ["C6", "C1"], ["C3", "N1"], ["C4", "N1"],
            ["N1", "O1"], ["N1", "O2"],
        ],
    },
    "reddit": {
        "name": "Reddit Ego-Thread",
        "nodes": [
            {"id": "OP", "x": 300, "y": 110, "feat": "C"},
            {"id": "U1", "x": 185, "y": 175, "feat": "C"},
            {"id": "U2", "x": 300, "y": 205, "feat": "C"},
            {"id": "U3", "x": 420, "y": 175, "feat": "N"},
            {"id": "U4", "x": 110, "y": 265, "feat": "O"},
            {"id": "U5", "x": 205, "y": 285, "feat": "O"},
            {"id": "U6", "x": 300, "y": 305, "feat": "O"},
            {"id": "U7", "x": 395, "y": 285, "feat": "O"},
            {"id": "U8", "x": 490, "y": 265, "feat": "O"},
            {"id": "U9", "x": 475, "y": 95, "feat": "O"},
        ],
        "edges": [
            ["OP", "U1"], ["OP", "U2"], ["OP", "U3"], ["OP", "U9"],
            ["U1", "U2"], ["U1", "U4"], ["U1", "U5"], ["U2", "U6"],
            ["U3", "U7"], ["U3", "U8"], ["U3", "U2"],
        ],
    },
}

RQ1_BENCHMARKS = [
    {"ds": "MUTAG", "type": "bio", "n": 188, "avgN": 18, "c": 2, "rwf": 87.7, "rival": 87.4, "who": "FGSD/NH", "delta": 0.3},
    {"ds": "PTC_MM", "type": "bio", "n": 336, "avgN": 14, "c": 2, "rwf": 67.2, "rival": 67.5, "who": "WL", "delta": -0.3},
    {"ds": "PTC_FR", "type": "bio", "n": 351, "avgN": 15, "c": 2, "rwf": 68.0, "rival": 67.9, "who": "WL", "delta": 0.1},
    {"ds": "PROTEINS", "type": "bio", "n": 1113, "avgN": 39, "c": 2, "rwf": 77.1, "rival": 76.1, "who": "WL", "delta": 1.0},
    {"ds": "DD", "type": "bio", "n": 1178, "avgN": 284, "c": 2, "rwf": 79.4, "rival": 79.1, "who": "WL", "delta": 0.3},
    {"ds": "IMDB-M", "type": "social", "n": 1500, "avgN": 13, "c": 3, "rwf": 49.6, "rival": 51.7, "who": "RWK", "delta": -2.1},
    {"ds": "RED-B", "type": "social", "n": 2000, "avgN": 430, "c": 2, "rwf": 90.7, "rival": 91.4, "who": "A-DOGE", "delta": -0.7},
    {"ds": "RED-5K", "type": "social", "n": 4999, "avgN": 509, "c": 5, "rwf": 55.6, "rival": 56.1, "who": "GIN", "delta": -0.5},
    {"ds": "DEE", "type": "social", "n": 9629, "avgN": 23, "c": 2, "rwf": 57.1, "rival": 56.9, "who": "GIN", "delta": 0.2},
    {"ds": "RED-12K", "type": "social", "n": 11929, "avgN": 391, "c": 11, "rwf": 48.1, "rival": 47.8, "who": "A-DOGE", "delta": 0.3},
    {"ds": "GIT", "type": "social", "n": 12725, "avgN": 114, "c": 2, "rwf": 70.4, "rival": 68.8, "who": "NetSimile", "delta": 1.6},
    {"ds": "TWI", "type": "social", "n": 127094, "avgN": 30, "c": 2, "rwf": 71.4, "rival": 70.5, "who": "NetSimile", "delta": 0.9},
]

RQ3_RUNTIMES = {
    "RED-B":   {"rwf": 20,   "netlsd": 221, "netsim": 312,  "fgsd": 1028},
    "RED-5K":  {"rwf": 56,   "netlsd": 509, "netsim": 599,  "fgsd": 1553},
    "RED-12K": {"rwf": 115,  "netlsd": 927, "netsim": 1094, "fgsd": 5872},
    "DEE":     {"rwf": 34,   "netlsd": 48,  "netsim": 175,  "fgsd": 2.7},
    "GIT":     {"rwf": 63,   "netlsd": 129, "netsim": 591,  "fgsd": 20},
    "TWI":     {"rwf": 2631, "netlsd": 422, "netsim": 2379, "fgsd": 43},
}
