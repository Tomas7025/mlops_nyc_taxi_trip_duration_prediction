#!/bin/bash

echo "🔍 Imagens que serão mantidas (mais recentes de cada repositório):"
echo "=================================================================="

REPOSITORIES=("serving" "training" "ghcr.io/tomas7025/proj_laia/serving" "ghcr.io/tomas7025/proj_laia/training")

for repo in "${REPOSITORIES[@]}"; do
    LATEST=$(docker images --format "{{.Repository}}:{{.Tag}} {{.CreatedAt}}" "$repo" | sort -r | head -n 1)
    if [ -n "$LATEST" ]; then
        echo "✅ $LATEST"
    fi
done

echo ""
echo "❗ Esta ação irá remover TODAS as outras imagens dos repositórios listados."
read -p "❓ Continuar? (s/N): " confirm

if [[ $confirm == [sS] ]]; then
    for repo in "${REPOSITORIES[@]}"; do
        echo "🧹 Limpando $repo..."
        docker images --format "{{.ID}} {{.Repository}}:{{.Tag}}" "$repo" | sort -r | tail -n +2 | awk '{print $1}' | xargs -r docker rmi -f
    done
    echo "🎉 Limpeza concluída!"
else
    echo "❌ Operação cancelada."
fi