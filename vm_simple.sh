#!/usr/bin/env zsh
# Solução SIMPLES para VMs LAIA - sem complicações

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Função simples: conecta a uma VM limpando known_hosts automaticamente
vm_connect() {
    local use_password=false
    local ip=""
    local user="admin"
    
    # Parse dos argumentos
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--password)
                use_password=true
                shift
                ;;
            *)
                if [[ -z "$ip" ]]; then
                    ip=$1
                elif [[ -z "$user" || "$user" == "admin" ]]; then
                    user=$1
                fi
                shift
                ;;
        esac
    done
    
    if [[ -z "$ip" ]]; then
        echo -e "${RED}Uso: vm_connect [-p|--password] <IP> [user]${NC}"
        echo -e "${BLUE}Exemplo: vm_connect 10.17.0.164${NC}"
        echo -e "${BLUE}Exemplo: vm_connect 10.17.0.164 laia${NC}"
        echo -e "${BLUE}Exemplo: vm_connect -p 10.17.0.164 laia  # usa password${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Conectando a $ip como $user...${NC}"
    
    # Remove IP antigo do known_hosts (evita conflitos)
    ssh-keygen -R "$ip" 2>/dev/null || true
    
    if [[ "$use_password" == true ]]; then
        echo -e "${BLUE}Modo password ativado (vai pedir password)${NC}"
        # Conecta com password (sem chave SSH)
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no "$user@$ip"
    else
        # Conecta com chave SSH (modo normal)
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/vms_dei_laia "$user@$ip"
    fi
}

# Função para limpar IPs antigos manualmente  
clean_ip() {
    local ip=$1
    if [[ -z "$ip" ]]; then
        echo -e "${RED}Uso: clean_ip <IP>${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Removendo $ip do known_hosts...${NC}"
    ssh-keygen -R "$ip"
    echo -e "${GREEN}Removido!${NC}"
}

# Função para mostrar histórico de IPs no known_hosts
show_vm_history() {
    echo -e "${BLUE}IPs de VMs no known_hosts:${NC}"
    grep "^10\.17\." ~/.ssh/known_hosts 2>/dev/null | awk '{print "  " $1}' | sort -u || echo "  Nenhum encontrado"
}

# Aliases simples
alias vmc='vm_connect'           # vmc 10.17.0.164
alias vmcl='vm_connect'          # vmcl 10.17.0.164 laia  
alias vmclean='clean_ip'         # vmclean 10.17.0.164
alias vmlist='show_vm_history'   # ver histórico

# Help
vm_help() {
    echo -e "${GREEN}=== VM Utils Simples ===${NC}"
    echo ""
    echo -e "${BLUE}Comandos principais:${NC}"
    echo "  vmc 10.17.0.164          # Conecta como admin (chave SSH)"
    echo "  vmc 10.17.0.164 laia     # Conecta como laia (chave SSH)"
    echo "  vmc -p 10.17.0.164       # Conecta como admin (password)"
    echo "  vmc -p 10.17.0.164 laia  # Conecta como laia (password)"
    echo "  vmclean 10.17.0.164      # Remove IP problemático"
    echo "  vmlist                   # Ver IPs no known_hosts"
    echo ""
    echo -e "${BLUE}Flags disponíveis:${NC}"
    echo "  -p, --password           # Força autenticação por password"
    echo ""
    echo -e "${BLUE}Como usar:${NC}"
    echo "1. Cria VM nova, nota o IP (ex: 10.17.0.200)"
    echo "2. Conecta: vmc 10.17.0.200"
    echo "3. Se der conflito: vmclean 10.17.0.200 && vmc 10.17.0.200"
    echo "4. Para usar password: vmc -p 10.17.0.200 laia"
}

# Alias para mostrar help quando necessário
alias vmhelp='vm_help'