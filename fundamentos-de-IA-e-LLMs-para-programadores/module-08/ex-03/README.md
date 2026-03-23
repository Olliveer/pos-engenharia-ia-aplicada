# Better Auth Demo

Demo simples de autenticação com Next.js + Better Auth + GitHub OAuth + SQLite.

## Pré-requisitos

- Node.js 18+
- Conta de desenvolvedor GitHub com OAuth App configurado

## Configuração do GitHub OAuth

1. Acesse [GitHub Developer Settings](https://github.com/settings/developers)
2. Clique em "New OAuth App"
3. Preencha:
   - **Application name**: Better Auth Demo
   - **Homepage URL**: `http://localhost:3000`
   - **Authorization callback URL**: `http://localhost:3000/api/auth/callback/github`
4. Copie o **Client ID**
5. Gere um **Client Secret**
6. Crie o arquivo `.env.local` na raiz do projeto:

```bash
cp .env.example .env.local
```

7. Edite `.env.local` com suas credenciais:

```
BETTER_AUTH_URL=http://localhost:3000
GITHUB_CLIENT_ID=seu_client_id
GITHUB_CLIENT_SECRET=seu_client_secret
```

## Instalação e Execução

```bash
# Instalar dependências
npm install

# Gerar tabelas do banco
echo 'y' | npx @better-auth/cli migrate

# Rodar em desenvolvimento
npm run dev
```

Acesse `http://localhost:3000` para ver o demo.

## Estrutura do Projeto

```
├── lib/
│   ├── auth.ts          # Configuração do Better Auth (server)
│   └── auth-client.ts   # Client para autenticação
├── app/
│   ├── api/auth/[...all]/route.ts  # API route do auth
│   ├── login/page.tsx   # Página de login
│   └── page.tsx         # Página home
├── better-auth.sqlite   # Banco SQLite gerado após migrate
└── .env.local           # Variáveis de ambiente (criar manualmente)
```
