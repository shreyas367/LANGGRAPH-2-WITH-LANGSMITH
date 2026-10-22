from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class state(TypedDict):
    a: int
    b: int
    c: int
    discrimant: float
    root1: float
    root2: float


def displa(state: state) -> state:
    print(f'{state["a"]}x^2 + {state["b"]}x + {state["c"]} = 0')
    return state


def discrimant(state: state) -> state:
    state["discrimant"] = (state["b"] ** 2) - 4 * state["a"] * state["c"]
    print(f'Discriminant: {state["discrimant"]}')
    return {"discrimant": state["discrimant"]}


def choise(state: state) -> str:
    if state["discrimant"] > 0:
        return "real root"
    elif state["discrimant"] == 0:
        return "equal root"
    else:
        return "complex root"


def real_root(state: state) -> state:
    r1 = (-state["b"] + (state["discrimant"] ** 0.5)) / (2 * state["a"])
    r2 = (-state["b"] - (state["discrimant"] ** 0.5)) / (2 * state["a"])
    print(f"Real roots: {r1}, {r2}")
    return {"root1": r1, "root2": r2}


def equal_root(state: state) -> state:
    r = -state["b"] / (2 * state["a"])
    print(f"Equal root: {r}")
    return {"root1": r, "root2": r}


def complex_root(state: state) -> state:
    print("Complex roots")
    return {}


graph = StateGraph(state)

graph.add_node("calculation", displa)
graph.add_node("discrimant", discrimant)
graph.add_node("real root", real_root)
graph.add_node("complex root", complex_root)
graph.add_node("equal root", equal_root)

graph.add_edge(START, "calculation")
graph.add_edge("calculation", "discrimant")

# Conditional edge based on choise
graph.add_conditional_edges("discrimant", choise)

graph.add_edge("real root", END)
graph.add_edge("complex root", END)
graph.add_edge("equal root", END)

app = graph.compile()

# Test

response=app.invoke({"a": 1, "b": -5, "c": 6})

print(response)
