# Harness VLA — Conclusões e Próximos Passos (execução na DGX)

Data: 2026-07-31 · Branch: `master` · Benchmark: EB-Manipulation, episódios fixos
[0, 15, 38] (pick-place / stack / wipe).

Este documento consolida onde chegamos com o loop de iteração paper-only
(Etapas A→E) e descreve a retomada na DGX. A meta é implementar o algoritmo da
v2 sem extensões e separar duas evidências: reprodução funcional com um backend
e protocolo usados pelo paper; e probes locais no EB-Manipulation, que não
reproduzem os resultados experimentais publicados.

---

## 1. Conclusão principal (fronteira paper-only)

Após 5 runs (Etapas A→E), o placar nos 3 episódios ficou estável em **1/3**, mas
as causas mudaram completamente. Os problemas de **formato e planejamento** foram
resolvidos; o que resta é **execução de contato**, que o paper delega ao VLA.

- **ep1 (pick-place): sucesso reprodutível** — 3 turnos, 6 steps, **0 parse_error**
  na Etapa E. Os mecanismos do paper que implementamos funcionam:
  thinking, memória global, failure models, reconciliação de estado, grounding e
  o vocabulário de primitivas (`move_to`, `move_pose`, `rotate_wrist`,
  `rotate_pitch`, `set_gripper`, `release`, `vla_act`).
- **ep2 (stack): falha de geometria (beta-only).** O planner faz tudo certo, mas o
  `vla_act place` analítico desce para `on.z = z_destino`
  ([primitives.py](../EmbodiedBench/embodiedbench/planner/harness/primitives.py#L665)),
  colocando o objeto no **mesmo nível** da base em vez de **em cima**. Nunca conta
  como stack.
- **ep3 (wipe): falha física de attach (categoria 6).** O corpo graspável é
  `sponge0` (collider ~58×125×33 mm), menor/deslocado do mesh visual
  `sponge_visual0` que o grounding publica. O attach depende de um proximity
  sensor estreito; a pose detectável exige offset 6-DoF específico
  (≈ +10.8, +37.3, +5.4 mm, orientação ≈ -93°). O grasp analítico desce ao
  centróide visual e nunca intercepta o collider.

**Verificação de fidelidade (`pesquisador-paper`):** o paper atribui contato e
placement constrained ao **VLA frozen**, mas mantém `release` como primitiva
analítica. Corrigir stack/wipe com heurísticas geométricas específicas continua
**beta-only** (compensa a ausência do backend publicado), portanto fora do
escopo "só o paper".
Conclusão: **com mecanismos exclusivamente do paper + o substituto analítico de
contato, o teto reprodutível é 1/3 (pick-place).** As outras duas tarefas exigem o
**VLA real**.

### Evolução das runs (episódios [0, 15, 38])

| Run | Sucessos | ep1 turnos | parse_errors | Falha residual dominante |
|---|---|---|---|---|
| baseline scripted | 1/3 | — | — | — |
| OpenVLA CPU | 0/3 | — | — | VLA inviável em CPU |
| Etapa A | 1/3 | 9 | 15 | formato/truncamento |
| Etapa C | 1/3 | 3 | 10 | estado + truncamento |
| Etapa D | 1/3 | 3 | ~6 | ruminação de false-success |
| Etapa E | 1/3 | 3 | ~6 (só ep2) | contato: geometria/attach (papel do VLA) |

Relatórios detalhados: `docs/runs/HARNESS_GEMMA_THINK_*.md`.
Vídeos: `videos_think/etapa{A,C,D,E}_episode_{1,2,3}.gif`.

---

## 2. Situação de hardware

Na máquina anterior (`4090nrc01`), a RTX 4090 não inicializava por
**mismatch de versão do driver**:

- Módulo do kernel **carregado**: `580.159.03` (antigo, residente na memória).
- Módulo DKMS no disco e biblioteca userspace: `580.173.02` (já batem entre si).
- `nvidia-smi` → *"Failed to initialize NVML: Driver/library version mismatch"*.

A correção exige recarregar o módulo do kernel (o `Xorg/gdm3` segura o GPU) **ou**
reiniciar — ambos precisam de **root**, ao qual não temos acesso nesta máquina.
Por isso a execução foi transferida para a DGX atual.

Estado verificado em 04/08/2026:

- host `dgx1`, 8× Tesla V100-SXM2 de 32 GB, compute capability 7.0;
- driver 550.144.03 e CUDA funcionais em container NVIDIA;
- PyTorch CUDA validado na imagem `nvcr.io/nvidia/pytorch:24.04-py3`;
- GPUs compartilhadas: selecionar novamente a GPU com mais memória livre e
   baixa utilização antes de iniciar cada processo;
- V100 requer FP16 e attention `eager`/`sdpa`; não usar BF16 nem
   FlashAttention 2;
- checkpoint, CoppeliaSim/PyRep, planner e dataset ainda precisam ser preparados
   nesta conta.

Correção de referência (na máquina com root), caso volte a ocorrer:

```bash
# Opção sem reboot (não afeta sessão VS Code remota):
sudo systemctl stop gdm3
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
nvidia-smi
# Opção simples: sudo reboot
```

Infra já presente e reutilizável: `nvidia-container-toolkit 1.18.1` instalado
(permite `docker run --gpus all`).

---

## 3. Próximos passos na máquina com GPU

Objetivo: fechar primeiro os contratos ausentes do algoritmo e então substituir
a física analítica de contato por um **VLA frozen**. A v2 usa πRLinf em LIBERO,
RLDX-1 em RoboCasa365 e LingBot-VLA em RoboTwin. OpenVLA aparece como baseline,
não como backend do Harness publicado.

1. **Validar CUDA no host** — `nvidia-smi` deve listar a GPU; conferir versão do
   driver == biblioteca userspace.
2. **Planner Gemma em GPU** — mover o Ollama/Gemma para GPU. Hoje em CPU custa
   ~126 s/turno; em GPU cai para segundos, viabilizando runs maiores.
   *(paper-compatible: só acelera o mesmo planner.)*
3. **Completar lifecycle, memória, REPL e percepção isolada** — integrar
   bootstrap/deployment, Task Specific Memory, Global Memory incremental,
   protocolo mediado por arquivos e RGB-D/world map sem poses oracle antes de
   declarar reprodução funcional. *(paper-confirmed.)*
4. **Servir um backend frozen publicado** — para a primeira reprodução funcional,
   o menor caminho da v2 é LIBERO + πRLinf, usando o mesmo checkpoint no baseline
   direto e no Harness. *(paper-confirmed.)*
5. **Manter OpenVLA apenas como probe EB-Manipulation** — o harness já tem o
   cliente pronto:
   [openvla_backend.py](../EmbodiedBench/embodiedbench/planner/harness/openvla_backend.py)
   (`OpenVLAHTTPBackend`, contrato: 1 delta-action de 7 valores por request,
   com `unnorm_key`). O checkpoint não está presente nesta máquina.
   *(beta-only como escolha de backend; adapter HTTP é paper-compatible.)*
6. **Ligar `vla_act` ao backend HTTP** — configurar o evaluator para usar o
   `OpenVLAHTTPBackend` no caminho de contato em vez da compilação analítica.
   Conferir o caminho "OpenVLA" vs "analítico" no
   `eb_manipulation_harness_evaluator.py`.
   *(paper-confirmed para o papel de `vla_act`; HTTP e conversão são locais.)*
7. **Container do simulador com GPU** — criar um runtime a partir deste checkout,
   sem usar o Dockerfile legado que clona o upstream em vez do código atual. O
   simulador pode permanecer desacoplado do VLA via HTTP.
8. **Re-rodar episódios [0, 15, 38]** como probe local e comparar contra a tabela da
   seção 1 (mesmos episódios, mesmo baseline). Métricas-chave: task_success por
   episódio, parse/compile rate, e se stack/wipe passam a completar.
9. **Documentar** a run em `docs/runs/` (timestamp, config, commit, índices) e
   arquivar GIFs em `videos_think/`.

### Riscos / atenção
- Nunca rodar duas instâncias do CoppeliaSim em paralelo.
- `openvla_cpu_eval/` (16 GB) permanece **gitignored**; sincronizar por outro meio
  (rsync/scp/HF) na máquina nova, não pelo git.
- Confirmar o `unnorm_key` correto do checkpoint (LIBERO) no
  `OpenVLAHTTPBackend`, senão o servidor retorna mismatch.
- Diferenciar sempre `action_success` de `task_success`.

---

## 4. Estado do repositório neste ponto

- Mecanismos do paper implementados e validados no planner (ver seção 1).
- Todos os relatórios de run e GIFs (Etapas A–E) versionados.
- Arquivos dos subagentes em `.github/agents/`.
- Correções desta sessão commitadas: reconciliação de estado, num_ctx efetivo,
  resiliência a timeout, failure model de grasp repetido, sinal de false-success
  no place. Ver `git log`.
