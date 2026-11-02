"""Pipeline stages. Each is a plain function run(cfg): it reads the previous stages' artifacts
from the work directory, writes its own and returns a dict of numbers for the manifest."""
