# -*- coding: utf-8 -*-
"""
Script: Ex Lab Mat.py

Descrição:
    Script para comparar três variantes do método de Newton (Clássico, Modificado e Discreto)
    para encontrar raízes de f(x) = 0. O código utiliza SymPy para manipulação simbólica,
    NumPy e Matplotlib para plotagem e Tabulate para exibir resultados em forma de tabela.

Bibliotecas principais e seus papéis:
    - sympy (sp): biblioteca de matemática simbólica. Usada para:
        * transformar a string da função em uma expressão simbólica (sympify),
        * calcular a derivada simbólica (diff),
        * converter expressões simbólicas em funções numéricas com lambdify.
    - tabulate: formata listas em tabelas legíveis no terminal (usado para sumarizar resultados).
    - math: fornece funções matemáticas básicas para lambdify em modo 'math' (uso numérico).
    - numpy (np): operações vetorizadas e criação de vetores para plotagem (linspace etc).
    - matplotlib.pyplot (plt): geração de gráficos da função para auxiliar a escolha de x0.

Observações gerais:
    - O script mantém um histórico (history) de iterações para cada método, contendo
      tuplas [k, xk, f(xk)] que permitem inspeção posterior.
    - Tratamentos de erro foram adicionados para lidar com falhas típicas:
      * falha na interpretação da função,
      * derivada nula,
      * não convergência dentro do número máximo de iterações.
"""

import sympy as sp           # manipulação simbólica (parsing, derivação)
from tabulate import tabulate  # exibição de resultados em tabela
import math                  # funções matemáticas para lambdify via 'math'
import numpy as np           # arrays numéricos e vetorização para plot
import matplotlib.pyplot as plt  # plotagem de gráficos


# --- 1. FUNÇÕES DE IMPLEMENTAÇÃO DOS MÉTODOS ---
# As três funções abaixo implementam variantes do método de Newton.
# Cada função retorna (raiz, history) onde history é uma lista com [k, xk, f(xk)].

def newton_classic(f, df, x0, tol, max_iter):
    """
    Resolve a raiz de uma função f(x) usando o Método de Newton Clássico.

    Parâmetros:
    f (sympy.Expr): A função simbólica f(x).
    df (sympy.Expr): A derivada simbólica f'(x).
    x0 (float): A estimativa inicial para a raiz.
    tol (float): A tolerância para o critério de parada (abs(f(xk)) < tol).
    max_iter (int): O número máximo de iterações.

    Retorna:
    tuple: (raiz (float) ou None, history (lista de iterações))
    """
    # Define o símbolo e cria funções numéricas a partir de expressões simbólicas.
    x = sp.symbols('x')
    try:
        # lambdify com 'math' gera funções escalares (compatíveis com float)
        f_num = sp.lambdify(x, f, 'math')
        df_num = sp.lambdify(x, df, 'math')
    except Exception as e:
        # Se a conversão falhar, reporta o erro e retorna sem histórico.
        print(f"Erro: {e}")
        return None, []

    xk, history = x0, []

    for k in range(max_iter):
        fx = f_num(xk)
        # Armazena o passo atual no histórico para posterior análise.
        history.append([k, xk, fx])

        # Critério de parada baseado no valor absoluto de f(xk).
        if abs(fx) < tol:
            return xk, history

        # Avalia a derivada no ponto atual. Verifica se é numericamente nula.
        dfx = df_num(xk)
        if abs(dfx) < 1e-15:
            # Evita divisão por zero ou passos instáveis quando a derivada é (quase) nula.
            print(f"Erro (Clássico): Derivada nula (f'({xk:.4f}) = 0).")
            return None, history

        # Passo padrão do Método de Newton.
        xk = xk - fx / dfx

    # Se exceder max_iter sem convergir, retorna falha com histórico.
    print(f"Erro (Clássico): Não convergiu após {max_iter} iterações.")
    return None, history


def newton_modified(f, df, x0, tol, max_iter):
    """
    Resolve a raiz usando o Método de Newton Modificado.
    A derivada é avaliada apenas na estimativa inicial x0 (economiza cálculos,
    mas pode reduzir a taxa de convergência).

    Parâmetros e retorno: mesmos que newton_classic.
    """
    x = sp.symbols('x')
    try:
        f_num = sp.lambdify(x, f, 'math')
        df_num = sp.lambdify(x, df, 'math')
    except Exception as e:
        print(f"Erro: {e}")
        return None, []

    xk, history = x0, []

    # Avalia a derivada uma única vez em x0.
    try:
        dfx0 = df_num(x0)
    except Exception as e:
        print(f"Erro: {e}")
        return None, []

    if abs(dfx0) < 1e-15:
        # Se a derivada inicial for zero, o método modificado não pode progredir.
        print(f"Erro (Modificado): Derivada inicial nula (f'({x0:.4f}) = 0).")
        return None, []

    for k in range(max_iter):
        fx = f_num(xk)
        history.append([k, xk, fx])

        if abs(fx) < tol:
            return xk, history

        # Note que aqui usamos dfx0 (derivada fixa), e não dfx atual.
        xk = xk - fx / dfx0

    print(f"Erro (Modificado): Não convergiu após {max_iter} iterações.")
    return None, history


def newton_discrete(f, x0, tol, max_iter, h=1e-8):
    """
    Resolve a raiz usando o Método de Newton Discreto.
    A derivada é aproximada por diferença finita:
      f'(x) ≈ (f(x + h) - f(x)) / h
    Isso evita a necessidade de derivada simbólica, porém depende do passo h.

    Parâmetros:
    f (sympy.Expr): A função simbólica f(x) (é convertida para função numérica).
    x0, tol, max_iter: como antes.
    h (float): passo pequeno para diferença finita (padrão 1e-8).

    Retorna:
    tuple: (raiz (float) ou None, history)
    """
    x = sp.symbols('x')
    try:
        f_num = sp.lambdify(x, f, 'math')
    except Exception as e:
        print(f"Erro: {e}")
        return None, []

    xk, history = x0, []

    for k in range(max_iter):
        fx = f_num(xk)
        history.append([k, xk, fx])

        if abs(fx) < tol:
            return xk, history

        # Avalia f(x+h) e calcula a derivada aproximada por diferença finita.
        try:
            f_xh = f_num(xk + h)
            df_discrete = (f_xh - fx) / h
        except ValueError:
            # Caso a função não aceite o ponto xk + h (ex.: domínio), captura o erro.
            print("Erro (Discreto): f(x+h).")
            return None, history

        if abs(df_discrete) < 1e-15:
            print(f"Erro (Discreto): Derivada nula (f'({xk:.4f}) \u2248 0).")
            return None, history

        xk = xk - fx / df_discrete

    print(f"Erro (Discreto): Não convergiu após {max_iter} iterações.")
    return None, history


# --- 2. FUNÇÃO PRINCIPAL E INTERFACE COM O USUÁRIO ---

def print_explanations():
    """
    Imprime explicações sobre os três métodos que serão comparados.
    Serve para orientar o usuário antes de solicitar entradas.
    """
    print("=" * 60)
    print("      COMPARAÇÃO DOS MÉTODOS DE NEWTON")
    print("=" * 60)
    print("\nEste script irá comparar três métodos para encontrar raízes (f(x) = 0):")
    print("\n### 1. Método de Newton Clássico ###")
    print("Fórmula: x_{k+1} = x_k - f(x_k) / f'(x_k)")
    print("\n### 2. Método de Newton Modificado ###")
    print("Fórmula: x_{k+1} = x_k - f(x_k) / f'(x_0)")
    print("\n### 3. Método de Newton Discreto ###")
    print("Fórmula: f'(x_k) \u2248 (f(x_k + h) - f(x_k)) / h")
    print("-" * 60)


# ===================================================================
# FUNÇÃO get_user_input MODIFICADA (AGORA COM PLOTAGEM NÃO-BLOQUEANTE)
# ===================================================================
def get_user_input():
    """
    Obtém a função f(x), sua derivada simbólica, estimativas iniciais, tolerância
    e número máximo de iterações do usuário. Gera também um gráfico da função
    em uma janela separada para ajudar na escolha dos palpites iniciais (x0).

    Retorna:
        f (sympy.Expr), df (sympy.Expr), x0_list (list de floats), tol (float), max_iter (int)
        ou (None, None, None, None, None) se ocorrer erro ao interpretar a função.
    """
    print("\nCONFIGURAÇÃO DO PROBLEMA:")

    # 1. Obter a função via input do usuário.
    str_f = input("Digite a função f(x) (ex: x**3 + x**2 - x): ")

    x = sp.symbols('x')
    try:
        # Transformar a string em expressão simbólica (sympify) e obter derivada.
        f = sp.sympify(str_f)
        df = sp.diff(f, x)
        print(f"\nFunção f(x): {f}")
        print(f"Derivada f'(x): {df}\n")
    except sp.SympifyError:
        print("Erro: Não foi possível entender a função. Tente usar a sintaxe do Python (ex: x**2).")
        return None, None, None, None, None

    # --- GERAR GRÁFICO DA FUNÇÃO (NÃO-BLOQUEANTE) ---
    print("\n--- Gerando o gráfico da sua função... ---")
    try:
        # lambdify com 'numpy' permite avaliar a função em vetores para plot.
        f_num_plot = sp.lambdify(x, f, 'numpy')

        x_vals = np.linspace(-5, 5, 400)
        y_vals = f_num_plot(x_vals)

        # Ativa o modo interativo do matplotlib para não bloquear a execução.
        plt.ion()
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {f}')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Eixo X (y=0)')
        plt.axvline(0, color='gray', linewidth=0.8, linestyle=':')
        plt.title(f'Gráfico da Função: {f}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()

        # Ajusta o limite vertical com base nos valores calculados (evita gráfico achatado).
        plt.ylim(min(y_vals) - 1, max(y_vals) + 1)

        # Mostra o gráfico sem bloquear a execução do restante do script.
        plt.show(block=False)
        plt.pause(0.1)  # Pequena pausa para garantir que a janela seja renderizada

        print("\n--- Gráfico exibido em uma janela separada. Deixe-a aberta para consulta! ---")
        print("Use o gráfico para identificar onde a função cruza o eixo X e escolher suas estimativas iniciais!\n")

    except Exception as e:
        # Se algum erro ocorrer na plotagem (ex.: domínio inválido para vetorização),
        # o script continua e solicita as estimativas normalmente.
        print(f"Erro ao gerar o gráfico: {e}")
        print("Continuando sem o gráfico. Por favor, forneça as estimativas iniciais.")

    # 2. Obter estimativas iniciais (x0). Pode-se fornecer múltiplas para tentar
    # encontrar várias raízes distintas.
    print("\n--- O que são as Estimativas Iniciais (x0)? ---")
    print("Para encontrar MÚLTIPLAS raízes, forneça MÚLTIPLAS estimativas.")

    x0_list = []
    while True:
        str_x0s = input("Digite as estimativas iniciais (separadas por vírgula, ex: -2, 0.1, 1): ")
        try:
            x0_list = [float(x.strip()) for x in str_x0s.split(',')]
            if not x0_list:
                raise ValueError("A lista não pode estar vazia.")
            break
        except ValueError as e:
            print(f"Entrada inválida. Certifique-se de usar números separados por vírgula. Erro: {e}")

    # 3. Obter tolerância (critério de parada).
    print("\n--- O que é o Nível de Tolerância (tol)? ---")
    while True:
        try:
            tol = float(input("Digite a tolerância (ex: 1e-7): "))
            if tol <= 0:
                print("A tolerância deve ser positiva.")
            else:
                break
        except ValueError:
            print("Entrada inválida.")

    # 4. Obter número máximo de iterações.
    print("\n--- O que é o Número Máximo de Iterações (max_iter)? ---")
    while True:
        try:
            max_iter = int(input("Digite o número máximo de iterações (ex: 100): "))
            if max_iter <= 0:
                print("O número de iterações deve ser positivo.")
            else:
                break
        except ValueError:
            print("Entrada inválida.")

    return f, df, x0_list, tol, max_iter
# ===================================================================
# FIM DA FUNÇÃO MODIFICADA
# ===================================================================


def main():
    """
    Função principal que orquestra a execução do programa:
      - Verifica disponibilidade das bibliotecas necessárias,
      - Imprime explicações, obtém entrada do usuário,
      - Executa os três métodos para cada estimativa inicial,
      - Mostra resumo em tabela e fecha os gráficos ao final.
    """
    try:
        sp; tabulate; np; plt
    except NameError:
        print("Erro: Bibliotecas (sympy, tabulate, numpy, matplotlib) não foram encontradas.")
        print("Instale-as com: pip install sympy tabulate numpy matplotlib")
        return

    print_explanations()

    f, df, x0_list, tol, max_iter = get_user_input()
    if f is None:
        # Se ocorreu erro ao interpretar a função, encerra.
        return

    print("=" * 60)
    print("             INICIANDO COMPARAÇÕES")
    print("=" * 60)

    for i, x0 in enumerate(x0_list):
        print(f"\n[ TENTATIVA {i + 1} / {len(x0_list)}: Iniciando com x0 = {x0} ]\n")

        summary_data = []

        # 1) Método Clássico
        print(f"--- 1. Clássico (x0 = {x0}) ---")
        root_classic, hist_classic = newton_classic(f, df, x0, tol, max_iter)
        if root_classic is not None:
            summary_data.append(["Clássico", root_classic, len(hist_classic) - 1, hist_classic[-1][2]])
        else:
            summary_data.append(["Clássico", "Falhou", "-", "-"])

        # 2) Método Modificado
        print(f"--- 2. Modificado (x0 = {x0}) ---")
        root_mod, hist_mod = newton_modified(f, df, x0, tol, max_iter)
        if root_mod is not None:
            summary_data.append(["Modificado", root_mod, len(hist_mod) - 1, hist_mod[-1][2]])
        else:
            summary_data.append(["Modificado", "Falhou", "-", "-"])

        # 3) Método Discreto
        print(f"--- 3. Discreto (x0 = {x0}) ---")
        root_disc, hist_disc = newton_discrete(f, x0, tol, max_iter, h=1e-8)
        if root_disc is not None:
            summary_data.append(["Discreto", root_disc, len(hist_disc) - 1, hist_disc[-1][2]])
        else:
            summary_data.append(["Discreto", "Falhou", "-", "-"])

        # Exibição do resumo em tabela. floatfmt controla a formatação numérica.
        print("\n" + ("-" * 60))
        print(f"    TABELA DE RESUMO (para x0 = {x0})")
        print(f"    Raízes encontradas (Próximas de {x0}):")
        print("-" * 60)
        headers = ["Método", "Raiz Encontrada", "Iterações (k)", "f(raiz)"]
        print(tabulate(summary_data, headers=headers, floatfmt=".10f"))
        print("_" * 60)

    print("\nCOMPARAÇÃO CONCLUÍDA.")

    # Mensagem indicando autoria (pedido específico).
    print("\nFeito por I.A")

    # Fecha todas as janelas de gráfico ao final.
    plt.close('all')


if __name__ == "__main__":
    main()
