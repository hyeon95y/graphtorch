REPO_ROOT=$(git rev-parse --show-toplevel)

# Install pre-commit
rm -f $REPO_ROOT/.git/hooks/pre-commit && rm -f $REPO_ROOT/.git/hooks/pre-commit.legacy
if [ -z ${VIRTUAL_ENV+x} ]; then
    echo "please run this script after activate virtualenv or conda."
    exit 1
fi
pip install pre-commit
cd $REPO_ROOT && pre-commit install
