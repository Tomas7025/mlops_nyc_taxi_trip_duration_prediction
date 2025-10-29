# 🧭 Guia de Setup — Projeto MLOps NYC Taxi (`proj_LAIA`)

Este documento explica **como configurar o ambiente de desenvolvimento e execução** do projeto,  
desde o acesso às VMs até à sincronização de dados e execução de código com tracking via MLflow.

---

## 🧩 1️⃣ Acesso à infraestrutura

### 🔹 Passo 1 — Conectar à rede interna
Para aceder às VMs, deves estar **ligado à rede interna do DEI** (ou VPN da Universidade de Coimbra).

- Se estás no campus: já estás dentro da rede interna.
- Se estás fora: liga-te à VPN institucional.

> ⚠️ Sem estar na rede interna, **não conseguirás aceder nem ao servidor DVC nem ao MLflow.**

---

### 🔹 Passo 2 — Acesso SSH à VM de treino

A VM principal (`ml-sv-laia-djtt`) contém:
- O *remote storage* do DVC (`/opt/dvc_storage`)
- O *MLflow Tracking Server* (porta `5000`)
- O ambiente de treino (Python, Docker, etc.)

Para aceder:
```bash
ssh admin@<IP_VM>
```

> 💡 Usa a tua chave pública SSH, que já deve estar configurada pelo administrador do projeto.

---

## 🧩 2️⃣ Clonar o repositório e criar ambiente local

No teu computador:

```bash
git clone git@github.com:Tomas7025/proj_LAIA.git
cd proj_LAIA
```

Cria e ativa o ambiente Python (recomendado usar **conda** ou **venv**):

### 👉 Se usas Conda:
```bash
conda create -n env_proj_laia python=3.12 -y
conda activate env_proj_laia
pip install -r requirements.txt
```

### 👉 Ou se usas venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> ⚠️ Se o `dvc pull` falhar mais à frente com erro de SSH, instala o plugin necessário:
> ```bash
> pip install dvc-ssh
> ```

---

## 🧩 3️⃣ Configurar o DVC

### 🔹 Passo 1 — Verificar instalação
```bash
dvc --version
```

Se não estiver instalado:
```bash
pip install dvc dvc-ssh
```

---

### 🔹 Passo 2 — Configurar o *remote storage* (na VM)

Primeiro, verifica se o repositório já tem um remote configurado:
```bash
dvc remote list
```

Se aparecer algo como:
```
local_storage   ssh://admin@<IP_VM>:/opt/dvc_storage   (default)
```
Então **atualiza a configuração existente**:
```bash
dvc remote modify --local local_storage url ssh://admin@<IP_VM>:/opt/dvc_storage
```

Ou, se tiveres o hostname configurado no `~/.ssh/config`:
```bash
dvc remote modify --local local_storage url ssh://my-vm-hostname:/opt/dvc_storage
```

Se **não aparecer nada**, adiciona o remote:
```bash
dvc remote add -d local_storage ssh://admin@<IP_VM>:/opt/dvc_storage
```

> 💡 O sufixo `--local` guarda a configuração apenas na tua máquina local (não é commitado no Git).

---

### 🔹 Passo 3 — Sincronizar os dados

Agora, puxa os dados do servidor DVC:
```bash
dvc pull -v -j 4
```

Este comando:
- Conecta-se à VM via SSH
- Faz download dos datasets necessários
- Cria a estrutura `data/raw/` localmente

Verifica:
```bash
ls -lh data/raw
```

Deves ver ficheiros como:
```
yellow_tripdata_2010-01.parquet
yellow_tripdata_2010-02.parquet
yellow_tripdata_2010-03.parquet
```

---

## 🧩 4️⃣ Acesso ao MLflow Tracking Server

O MLflow está configurado **para uso interno na VM** — ou seja, apenas pode ser acedido:
- diretamente dentro da VM (localhost)
- ou via túnel SSH se estiveres remoto na VPN

---

### 🔹 Aceder dentro da VM

Na VM (`admin@ml-sv-laia-djtt`), o MLflow já está a correr via Docker Compose:
```bash
cd ~/proj_LAIA/infra/mlflow_docker
docker compose ps
```

Se não estiver a correr:
```bash
docker compose up -d
```

Verifica se responde:
```bash
curl -v http://127.0.0.1:5000
```

Abre o servidor MLflow com o ip da **VM** no teu browser (assumindo que estás na mesma rede interna que a VM):
```
http://<IP_VM>:5000
```

---

### 🔹 Aceder remotamente (via túnel SSH)

Se estás **fora da VM** mas ligado à VPN:
```bash
ssh -L 5000:localhost:5000 admin@<IP_VM>
```

Depois, no teu browser local:
```
http://127.0.0.1:5000
```

> Isto cria um *túnel SSH* que redireciona a porta `5000` da VM para a tua máquina local.

---

### 🔹 Utilizar o MLflow nos teus scripts

Adiciona no início do teu código (ex: `src/train.py`):
```python
import mlflow

# Se estiveres dentro da VM:
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Ou se estiveres a usar túnel SSH (fora da VM, mas na VPN):
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("nyc_taxi_project")
```

Cada execução de treino registará:
- parâmetros do modelo
- métricas
- artefactos (ficheiros, checkpoints, etc.)

Se não quiseres que certas execuções apareçam no MLflow (por exemplo, testes locais),  
podes condicionar o tracking:
```python
import os
import mlflow

if os.getenv("USE_MLFLOW", "true").lower() == "true":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.start_run()
```
E desativar com:
```bash
export USE_MLFLOW=false
```

---

## 🧩 5️⃣ Estrutura do projeto

```
proj_LAIA/
├── data/                     # Dados versionados (via DVC)
│   ├── raw/
│   └── .gitkeep
├── infra/
│   ├── server_ml_laia.yaml   # Cloud-init: setup automático da VM
│   ├── setup_oh_my_zsh.sh    # Configuração shell (Zsh, plugins)
│   ├── vm_simple.sh          # Criação de VM
│   └── mlflow_docker/        # MLflow Tracking Server (Docker)
│       ├── docker-compose.yml
│       └── mlflow.db
├── notebooks/                # Análise exploratória
├── src/                      # Código fonte
│   ├── train.py
│   └── utils/
│       └── preprocessing.py
├── requirements.txt          # Dependências
└── dvc.yaml                  # (futuramente) pipeline de treino
```

---

## 🧩 6️⃣ Como correr experimentos (exemplo rápido)

Na VM (onde o MLflow está ativo):
```bash
python src/train.py
```

Isso irá:
- Usar os dados de `data/raw/`
- Treinar o modelo
- Registar os resultados no MLflow

Depois, abre no browser da VM ou via túnel:
```
http://127.0.0.1:5000
```
para ver os resultados.

---

## 🧩 7️⃣ Dicas e boas práticas

✅ **Nunca commits ficheiros grandes** — usa o DVC para dados  
✅ **Não mexer em `/opt/dvc_storage` diretamente**  
✅ **Antes de treinar um modelo:** faz `git pull` e `dvc pull`  
✅ **Usa branches** para alterações grandes  
✅ **Guarda tokens e chaves em variáveis de ambiente ou GitHub Secrets**

---

## 🧩 8️⃣ Problemas comuns

| Problema | Solução |
|-----------|----------|
| `dvc pull` dá erro de cache | Instala `dvc-ssh` e verifica a VPN |
| MLflow não abre | Confirma que o container está ativo (`docker compose ps`) |
| Docker erro “permission denied” | Tenta `sudo docker compose up -d` |
| `.dvc` não trackeia | Garante que `.gitignore` tem `!data/**/*.dvc` |
| `Remote already exists` | Usa `dvc remote modify --local local_storage url ...` |

---

## 🧠 9️⃣ Recursos úteis

- 📘 [Documentação DVC](https://dvc.org/doc/start)
- 📗 [Documentação MLflow](https://mlflow.org/docs/latest/index.html)
- 🐋 [Docker Compose](https://docs.docker.com/compose/)
- 💡 [Git SSH Keys Setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

---
