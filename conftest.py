"""Assure que la racine du dépôt est sur sys.path pour `import functions...`
quel que soit le répertoire depuis lequel pytest est lancé.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
