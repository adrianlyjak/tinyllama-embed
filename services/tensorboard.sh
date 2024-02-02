cd "$(dirname $0)/.."

if [ -z "$1" ]; then
    echo "Usage: $0 <log_dir>"
    exit 1
fi

docker run \
    -it --rm \
    -v "$PWD/$log_dir":/app/runs/ \
    -p 6006:6006 \
    -w "/app/" \
    --name "tensorboard" \
    schafo/tensorboard