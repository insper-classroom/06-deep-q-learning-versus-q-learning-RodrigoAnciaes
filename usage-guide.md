# Comparação de RL para Mountain Car - Guia de Uso Rápido

## Requisitos
```
pip install gymnasium matplotlib numpy torch tqdm
```

## Uso Básico

### Treinar novos agentes:
```
python compare.py --runs 5 --episodes 5000
```

### Carregar resultados previamente treinados:
```
python compare.py --load
```

### Testar agentes treinados:
```
python compare.py --test --algorithm both --test_run 0 --render
```

### Plotar resultados dos testes dos agentes treinados:
```
python compare.py --test --algorithm both --test_episodes 1000 --test_run 0 --plot-test
```

## Argumentos do Comando
- `--runs N`: Número de execuções de treinamento por algoritmo
- `--episodes N`: Número máximo de episódios de treinamento por execução
- `--seed N`: Definir semente aleatória
- `--algorithm [q_learning|dqn|both]`: Escolher algoritmo(s)
- `--test`: Testar em vez de treinar
- `--render`: Visualizar o ambiente durante os testes
- `--test_episodes N`: Número de episódios de teste
- `--test_run N`: ID específico de execução para testar
- `--plot-test`: Plotar resultados dos testes

## Saída
Os resultados são salvos no diretório `comparison/`, incluindo curvas de aprendizado e estatísticas de desempenho.
