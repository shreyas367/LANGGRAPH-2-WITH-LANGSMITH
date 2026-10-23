from __future__ import annotations
from fastmcp import FastMCP
import math


mcp=FastMCP("Arithmatic")

# def _as_number(x):

#     if isinstance(x,(int,float)):
#         return float(x)
#     if isinstance(x,str):
#         return float(x.strip())
#     raise ValueError(f"{x} is not a number")

@mcp.tool()
async def add(a:float,b:float)->float:
    return a+b

@mcp.tool()
async def sub(a:float,b:float)->float:
    return a-b

@mcp.tool()
async def mul(a:float,b:float)->float:
    return a*b

@mcp.tool()
async def div(a:float,b:float)->float:
    return a/b

@mcp.tool()
async def sqrt(x:float)->float:
    return math.sqrt(x)

if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)


