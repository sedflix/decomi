#! /bin/bash

result=$(find ./divided_wiki/en -name '*bz2' -exec bzcat {} \+ \
        | pv \
        | tee >(    sed 's/<[^>]*>//g' \
                  | sed 's|["'\''„“‚‘]||g' \
                  | python3 ./wiki_cleaner.py en >> wiki_en.txt \
               ) \
        | grep -e "<doc" \
        | wc -l)

echo "$result"
