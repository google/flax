set -e

cd ../../..
source .venv/bin/activate
cd flax/experimental/nnx

for f in $(find examples -name "*.py" -maxdepth 1); do
    echo -e "\n---------------------------------"
    echo "$f"
    echo "---------------------------------"
    MPLBACKEND=Agg python "$f"
done
