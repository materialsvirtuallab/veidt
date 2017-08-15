from invoke import task
import os
import webbrowser
import datetime

from monty.os import cd

"""
Deployment file to facilitate releases of veidt.
Note that this file is meant to be run from the root directory of the veidt
repo.
"""

__author__ = "Shyue Ping Ong"
__email__ = "ongsp@ucsd.edu"
__date__ = "Sep 1, 2014"


@task
def make_doc(ctx):

    with cd("docs_rst"):
        ctx.run("sphinx-apidoc --separate -d 6 -o . -f ../veidt")
        ctx.run("rm veidt*.tests.*rst")
        ctx.run("make html")
        ctx.run("cp _static/* ../docs/html/_static", warn=True)

    with cd("docs"):
        ctx.run("cp -r html/* .")
        ctx.run("rm -r html")
        ctx.run("rm -r doctrees")
        ctx.run("rm -r _sources")

        # This makes sure veidt.org works to redirect to the Gihub page
        # Avoid the use of jekyll so that _dir works as intended.
        ctx.run("touch .nojekyll")

@task
def update_doc(ctx):
    make_doc(ctx)
    ctx.run("git add .")
    ctx.run("git commit -a -m \"Update dev docs\"")
    ctx.run("git push")


@task
def open_doc(ctx):
    pth = os.path.abspath("docs/index.html")
    webbrowser.open("file://" + pth)
