@echo off
REM This batch script runs all search methods for a given map file.

SET FILENAME=PathFinder-test.txt.txt

echo Running BFS...
python search.py %FILENAME% bfs

echo Running DFS...
python search.py %FILENAME% dfs

echo Running UCS...
python search.py %FILENAME% ucs

echo Running GBFS...
python search.py %FILENAME% gbfs

echo Running A*...
python search.py %FILENAME% as

echo Running Custom Search 1 (UCS)...
python search.py %FILENAME% cus1

echo Running Custom Search 2 (IDDFS)...
python search.py %FILENAME% cus2

echo All methods completed.
pause
