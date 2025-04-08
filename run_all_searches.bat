@echo off
REM This batch script runs all search methods for a given map file.

SET FILENAME=PathFinder-test.txt

echo Running BFS...
python search.py PathFinder-test.txt bfs

echo Running DFS...
python search.py PathFinder-test.txt dfs

echo Running IDDFS...
python search.py PathFinder-test.txt iddfs

echo Running UCS...
python search.py PathFinder-test.txt ucs

echo Running GBFS...
python search.py PathFinder-test.txt gbfs

echo Running A*...
python search.py PathFinder-test.txt as

echo All methods completed.
pause
