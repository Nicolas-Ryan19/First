# 🔬 Comparação dos Métodos de Newton-Raphson (Clássico, Modificado e Discreto)

Este repositório contém um script Python para demonstrar e comparar a eficiência, a taxa de convergência e as características de três variações do popular Método de Newton (ou Newton-Raphson) para encontrar raízes de funções $f(x) = 0$.

## 🎯 Objetivo

O objetivo principal do script é permitir que o usuário insira uma função qualquer, defina estimativas iniciais e observe o comportamento e a performance de cada método em relação à velocidade de convergência e estabilidade.

## ⚙️ Métodos Implementados

O script implementa e compara as seguintes variações do método de Newton:

| Método | Fórmula de Iteração | Derivada ($f'(x)$) | Vantagens |
| :--- | :--- | :--- | :--- |
| **1. Newton Clássico** | $$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$$ | Calculada e atualizada a cada iteração ($f'(x_k)$). | Convergência quadrática (muito rápida) sob boas condições. |
| **2. Newton Modificado** | $$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_0)}$$ | Calculada apenas uma vez na estimativa inicial ($f'(x_0)$). | Requer menos cálculos por iteração, ideal para funções com derivadas complexas. |
| **3. Newton Discreto** | $$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$$| Aproximada usando diferenças finitas:$$f'(x_k) \approx \frac{f(x_k + h) - f(x_k)}{h}$$ | Não requer o cálculo da derivada analítica ($f'(x)$), útil para funções onde a derivada é desconhecida ou difícil de obter. |

## 🚀 Como Executar o Script

### Pré-requisitos

O script utiliza as seguintes bibliotecas Python, que devem ser instaladas no seu ambiente:

* `sympy`: Para manipulação simbólica de funções e derivadas.
* `tabulate`: Para exibir os resultados em formato de tabela limpa no console.
* `numpy`: Para cálculos numéricos (usado na plotagem).
* `matplotlib`: Para gerar o gráfico auxiliar da função.

Você pode instalar todas as dependências com o seguinte comando:

```bash
pip install sympy tabulate numpy matplotlib
