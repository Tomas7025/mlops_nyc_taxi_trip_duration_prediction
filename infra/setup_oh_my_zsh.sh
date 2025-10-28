#!/usr/bin/env bash
set -euo pipefail

# setup_oh_my_zsh.sh
# Idempotent installer for zsh + oh-my-zsh + common plugins
# Usage examples:
#   sudo ./setup_oh_my_zsh.sh --user admin --user laia
#   sudo ./setup_oh_my_zsh.sh --users admin,laia --powerlevel10k
# Notes:
# - Run as root (sudo) so it can install packages and write to user home dirs.
# - It will not overwrite an existing ~/.zshrc; it will append or modify the plugins line.
# - Plugins installed to $HOME/.oh-my-zsh/custom/plugins for each user.

USERS=()
INSTALL_POWERLEVEL10K=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      shift; USERS+=("$1") ;;
    --users)
      shift; IFS=',' read -ra ADDR <<< "$1"; for u in "${ADDR[@]}"; do USERS+=("$u"); done ;;
    --powerlevel10k)
      INSTALL_POWERLEVEL10K=true ;;
    -h|--help)
      sed -n '1,160p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

if [[ ${#USERS[@]} -eq 0 ]]; then
  echo "No users specified. Use --user or --users. Example: --user admin --user laia" >&2
  exit 1
fi

# Package install helper (Debian/Ubuntu)
install_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y zsh git curl fzf fonts-powerline || true
  else
    echo "Package manager apt-get not found; please install zsh, git, curl manually." >&2
  fi
}

install_packages

for user in "${USERS[@]}"; do
  # Resolve user info
  if ! id "$user" >/dev/null 2>&1; then
    echo "User $user does not exist; skipping" >&2
    continue
  fi
  USER_HOME=$(eval echo "~$user")
  echo "Setting up oh-my-zsh for $user (home: $USER_HOME)"

  # Ensure zsh is the shell for the user
  if ! grep -q "$(which zsh)" /etc/shells; then
    echo "$(which zsh)" | tee -a /etc/shells
  fi
  chsh -s "$(which zsh)" "$user" || true

  # Install oh-my-zsh non-interactively if not present
  if [[ ! -d "$USER_HOME/.oh-my-zsh" ]]; then
    sudo -u "$user" env RUNZSH=no CHSH=no KEEP_ZSHRC=yes sh -c \
      "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
  else
    echo "oh-my-zsh already installed for $user"
  fi

  # Create custom plugins dir var for user
  ZSH_CUSTOM="$USER_HOME/.oh-my-zsh/custom"
  PLUGINS=(zsh-syntax-highlighting zsh-autosuggestions zsh-completions zsh-history-substring-search)

  for p in "${PLUGINS[@]}"; do
    if [[ ! -d "$ZSH_CUSTOM/plugins/$p" ]]; then
      case "$p" in
        zsh-syntax-highlighting)
          sudo -u "$user" git clone https://github.com/zsh-users/zsh-syntax-highlighting.git "$ZSH_CUSTOM/plugins/$p"
          ;;
        zsh-autosuggestions)
          sudo -u "$user" git clone https://github.com/zsh-users/zsh-autosuggestions.git "$ZSH_CUSTOM/plugins/$p"
          ;;
        zsh-completions)
          sudo -u "$user" git clone https://github.com/zsh-users/zsh-completions.git "$ZSH_CUSTOM/plugins/$p"
          ;;
        zsh-history-substring-search)
          sudo -u "$user" git clone https://github.com/zsh-users/zsh-history-substring-search.git "$ZSH_CUSTOM/plugins/$p"
          ;;
      esac
    else
      echo "Plugin $p already installed for $user"
    fi
  done

  # Optionally install powerlevel10k
  if [[ "$INSTALL_POWERLEVEL10K" == true ]]; then
    if [[ ! -d "$ZSH_CUSTOM/themes/powerlevel10k" ]]; then
      sudo -u "$user" git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "$ZSH_CUSTOM/themes/powerlevel10k"
    else
      echo "powerlevel10k already installed for $user"
    fi
  fi

  # Update .zshrc: add/replace plugins line and add fpath line for zsh-completions
  ZSHRC="$USER_HOME/.zshrc"
  # Ensure file exists
  if [[ ! -f "$ZSHRC" ]]; then
    sudo -u "$user" cp "$USER_HOME/.oh-my-zsh/templates/zshrc.zsh-template" "$ZSHRC" || true
  fi

  # Backup
  cp "$ZSHRC" "$ZSHRC.bak-$(date +%Y%m%d%H%M%S)"

  # Replace plugins line
  if grep -q '^plugins=' "$ZSHRC"; then
    sed -i "s/^plugins=.*/plugins=(git docker zsh-completions zsh-history-substring-search zsh-autosuggestions zsh-syntax-highlighting)/" "$ZSHRC"
  else
    echo "plugins=(git docker zsh-completions zsh-history-substring-search zsh-autosuggestions zsh-syntax-highlighting)" >> "$ZSHRC"
  fi

  # Ensure fpath line exists just after plugins line
  if ! grep -q 'zsh-completions' "$ZSHRC"; then
    awk '/^plugins=/{print; print "fpath=(\$HOME/.oh-my-zsh/custom/plugins/zsh-completions/src \$fpath)"; next}1' "$ZSHRC" > "$ZSHRC.tmp" && mv "$ZSHRC.tmp" "$ZSHRC"
  fi

  # If powerlevel10k requested, set theme
  if [[ "$INSTALL_POWERLEVEL10K" == true ]]; then
    sed -i "s/^ZSH_THEME=.*/ZSH_THEME='powerlevel10k\/powerlevel10k'/" "$ZSHRC" || true
  fi

  # Ensure essential oh-my-zsh init lines exist (export ZSH, ZSH_THEME, source $ZSH/oh-my-zsh.sh)
  if ! grep -q '^export ZSH=' "$ZSHRC"; then
    echo 'export ZSH="$HOME/.oh-my-zsh"' | sudo -u "$user" tee -a "$ZSHRC" >/dev/null
  fi

  if ! grep -q '^ZSH_THEME=' "$ZSHRC"; then
    if [[ "$INSTALL_POWERLEVEL10K" == true ]]; then
      echo "ZSH_THEME='powerlevel10k/powerlevel10k'" | sudo -u "$user" tee -a "$ZSHRC" >/dev/null
    else
      echo "ZSH_THEME='robbyrussell'" | sudo -u "$user" tee -a "$ZSHRC" >/dev/null
    fi
  fi

  if ! grep -q 'oh-my-zsh.sh' "$ZSHRC"; then
    echo 'source $ZSH/oh-my-zsh.sh' | sudo -u "$user" tee -a "$ZSHRC" >/dev/null
  fi

  # Fix ownership
  chown -R "$user:$user" "$USER_HOME/.oh-my-zsh" || true
  chown "$user:$user" "$ZSHRC" || true

  echo "Done for $user"

done

# Mark todo step completed
if command -v python3 >/dev/null 2>&1; then
  echo "" 
fi

exit 0
