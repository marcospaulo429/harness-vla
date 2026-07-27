# Harness beta: contrato bootstrap/deployment

## Classificacao

**paper-confirmed:** a Secao 2.2 e o Apendice C de
`arXiv:2607.08448v2` separam uma seed de bootstrap das seeds de deployment.
Bootstrap permite exploracao, reset e escrita de Task Specific Memory; deployment
usa seeds held-out, budget menor, reset desabilitado e memoria read-only. A seed
de bootstrap nao entra nas metricas reportadas.

**beta-only:** schema Python, enums, mensagens de erro e serializacao do
manifesto.

## Implementacao

Commit: `3cdf373`.

O contrato puro define:

- fases `bootstrap` e `deployment`;
- operacoes guardadas de reset, leitura/escrita de memoria e reporte;
- seeds de bootstrap e avaliacao obrigatoriamente disjuntas;
- bootstrap nao reportavel;
- deployment sem reset e sem escrita de memoria;
- budget de bootstrap maior ou igual ao de deployment;
- manifesto deterministico com permissoes derivadas, nao configuraveis.

O protocolo local `[0]` para bootstrap e `[15,38]` para deployment foi usado
como fixture. O guard rejeita tanto reportar a seed 0 quanto executar uma seed na
fase errada.

## Validacao

- 150 testes passaram;
- diagnosticos do editor limpos;
- `git diff --check` passou;
- nenhuma mudanca no evaluator ou no comportamento das runs existentes.

## Limite

Esta etapa implementa o contrato, mas ainda nao o conecta ao evaluator. Portanto
nenhuma run fisica pode ser declarada bootstrap/deployment protegida por ele. A
integracao futura deve guardar reset, escrita de memoria e agregacao de metricas
no ponto em que cada operacao ocorre, e persistir o manifesto na run.
