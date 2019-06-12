#! /bin/bash

set -ex


upload_coverage() {
    codecov
}


upload_docs() {
    pushd "docs/build/html"
    if [[ ! -z "$(git status --porcelain)" ]]; then
        git add "."
        git commit --amend --message "Travis build: ${TRAVIS_BUILD_NUMBER}"
        git remote add origin-pages "https://${GITHUB_TOKEN}@github.com/grand-mother/grand-docs.git" > /dev/null 2>&1
        git push -f --quiet --set-upstream origin-pages master
    fi
    popd
}


upload_coverage
upload_docs
