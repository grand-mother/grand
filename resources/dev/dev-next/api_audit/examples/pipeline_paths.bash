# Run at the root of a checkout. Mirrors how bintoroot.bash finds the repository.
for script_dir in "$PWD/scripts/transfers" "$PWD/scripts/pipeline"; do
  [ -e "$script_dir/bintoroot.bash" ] || continue
  grand_dir=${script_dir%/scripts/transfers}          # line 6 of bintoroot.bash
  root=${grand_dir#$PWD}
  echo "bintoroot.bash is in ${script_dir#$PWD/}; it takes the repository to be ${root:-the checkout root}"
  for f in scripts/transfers/register_convert.py granddb/register_file_in_db.py; do
    [ -e "$grand_dir/$f" ] && echo "  found   $f" || echo "  missing $f, looked for at ${grand_dir#$PWD/}/$f"
  done
done
echo "paths still used by scripts/transfers:"
grep -ho "softs/grand/scripts/transfers/[a-z_]*\.[a-z]*" scripts/transfers/*.bash | sort -u | while read -r p; do
  f=${p#softs/grand/}
  [ -e "$f" ] && echo "  found   $f" || echo "  missing $f"
done
