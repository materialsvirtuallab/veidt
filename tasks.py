from invoke import task
import os
import webbrowser
import re
import json
import requests
from monty.os import cd
from veidt import __version__ as VERSION

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


@task
def publish(ctx):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py register sdist bdist_wheel")
    ctx.run("twine upload dist/*")

@task
def release_github(ctx):
    payload = {
        "tag_name": "v" + VERSION,
        "target_commitish": "master",
        "name": "v" + VERSION,
        "body": "Release",
        "draft": False,
        "prerelease": False
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallalb/veidt/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


@task
def release(ctx, notest=False):
    if not notest:
        ctx.run("nosetests")
    publish(ctx)
    update_doc(ctx)
    ctx.run("git tag -a v%s -m \"v%s release\"" % (VERSION, VERSION))
    ctx.run("git push --tags")
    release_github(ctx)
