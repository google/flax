set -e

cd ../../..
source .venv/bin/activate
cd flax/experimental/nnx

for f in $(find examples -name "*.py"); do
    echo -e "\n---------------------------------"
    echo "$f"
    echo "---------------------------------"
    python "$f"
done
