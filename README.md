# AI Code Reviewer using CrewAI and GitHub Webhooks

Este projeto implementa um serviço de revisão de código automatizado que utiliza agentes de IA (CrewAI com LLMs da OpenAI) para analisar Pull Requests no GitHub e postar comentários de revisão diretamente no PR.

## Funcionalidades

- Recebe eventos de Pull Request (`opened`, `synchronize`, `reopened`) via Webhooks do GitHub.
- Verifica a assinatura do webhook para segurança.
- Filtra arquivos modificados para analisar apenas código relevante (ignora binários, lockfiles, etc.).
- Utiliza um Agente Analisador (CrewAI) para identificar problemas de qualidade, segurança, performance e estilo nas linhas de código _adicionadas_.
- Utiliza um Agente Comentador (CrewAI) para:
  - Traduzir o feedback para Português do Brasil (pt-BR).
  - Encontrar a posição exata da linha no diff.
  - Postar comentários inline no GitHub Review ou comentários gerais no PR.
- Posta um resumo final da análise no PR.
- Configurável via variáveis de ambiente.

## Configuração

### 1. Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto ou defina as seguintes variáveis de ambiente:

- `GITHUB_TOKEN`: Um Personal Access Token (PAT) do GitHub com permissões `repo` (ou `public_repo` para repositórios públicos) e `pull_requests:write`.
- `OPENAI_API_KEY`: Sua chave de API da OpenAI.
- `GITHUB_WEBHOOK_SECRET`: Um segredo forte e aleatório que você configurará no webhook do GitHub. Use um gerador de senhas.
- `OPENAI_MODEL_NAME` (Opcional): O nome do modelo OpenAI a ser usado (padrão: "gpt-4-turbo"). Verifique os modelos disponíveis na sua conta OpenAI.
- `PORT` (Opcional): A porta em que o servidor Flask será executado (padrão: 5001).

### 2. Ambiente Python

Recomenda-se usar um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate    # Windows
```
