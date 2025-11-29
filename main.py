# Intentionally left blank
import click

@click.group()
@click.version_option("1.0.0", "-v", "--version", help="Show version and exit.")
def cli():
    """
    CLI for the project [Easy Access]
    """
    pass