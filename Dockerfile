FROM ubuntu:24.04@sha256:562111b88caa79c71391c0c4d3c2114a7d5c35915c20561fd893cb1e07929e6b AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake g++ git ninja-build && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY patches/ patches/
COPY scripts/build.sh scripts/build.sh
RUN JOBS=4 scripts/build.sh

FROM ollama/ollama:0.40.18@sha256:1cabd20e1762a7a8444bf4a7110b346d923cd77ab6c97c984434140638c913ec AS ollama

FROM ubuntu:24.04@sha256:562111b88caa79c71391c0c4d3c2114a7d5c35915c20561fd893cb1e07929e6b
COPY --from=ollama /bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama/GO_LICENSE /usr/lib/ollama/GO_LICENSE
COPY --from=build /src/bin/llama-server /usr/lib/ollama/llama-server
COPY --from=build /src/bin/llama-completion /usr/bin/llama-completion
COPY LICENSE patches/llama.cpp-LICENSE notices/ollama-LICENSE /usr/share/doc/littlebit/
WORKDIR /opt/littlebit
COPY Modelfile .
ENV OLLAMA_HOST=0.0.0.0:11434 OLLAMA_NO_CLOUD=1
EXPOSE 11434
ENTRYPOINT ["ollama"]
CMD ["serve"]
