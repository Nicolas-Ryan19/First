import sympy as sp
from tabulate import tabulate
import math
import numpy as np
import matplotlib.pyplot as plt

# --- 1. FUNÇÕES DE IMPLEMENTAÇÃO DOS MÉTODOS ---
# (Cole as 3 funções: newton_classic, newton_modified, newton_discrete
#  exatamente como estavam no script anterior. Elas não mudam.)

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
    tuple: Uma tupla contendo a raiz encontrada (float) ou None se não convergir,
           e uma lista (history) com o histórico de iterações.
    """
    x = sp.symbols('x')
    try: f_num = sp.lambdify(x, f, 'math'); df_num = sp.lambdify(x, df, 'math')
    except Exception as e: print(f"Erro: {e}"); return None, []
    xk, history = x0, []
    for k in range(max_iter):
        fx = f_num(xk); history.append([k, xk, fx])
        if abs(fx) < tol: return xk, history
        dfx = df_num(xk)
        if abs(dfx) < 1e-15: print(f"Erro (Clássico): Derivada nula (f'({xk:.4f}) = 0)."); return None, history
        xk = xk - fx / dfx
    print(f"Erro (Clássico): Não convergiu após {max_iter} iterações."); return None, history

def newton_modified(f, df, x0, tol, max_iter):
    """
    Resolve a raiz de uma função f(x) usando o Método de Newton Modificado.
    A derivada é avaliada apenas na estimativa inicial x0.

    Parâmetros:
    f (sympy.Expr): A função simbólica f(x).
    df (sympy.Expr): A derivada simbólica f'(x).
    x0 (float): A estimativa inicial para a raiz.
    tol (float): A tolerância para o critério de parada (abs(f(xk)) < tol).
    max_iter (int): O número máximo de iterações.

    Retorna:
    tuple: Uma tupla contendo a raiz encontrada (float) ou None se não convergir,
           e uma lista (history) com o histórico de iterações.
    """
    x = sp.symbols('x')
    try: f_num = sp.lambdify(x, f, 'math'); df_num = sp.lambdify(x, df, 'math')
    except Exception as e: print(f"Erro: {e}"); return None, []
    xk, history = x0, []
    try: dfx0 = df_num(x0)
    except Exception as e: print(f"Erro: {e}"); return None, []
    if abs(dfx0) < 1e-15: print(f"Erro (Modificado): Derivada inicial nula (f'({x0:.4f}) = 0)."); return None, []
    for k in range(max_iter):
        fx = f_num(xk); history.append([k, xk, fx])
        if abs(fx) < tol: return xk, history
        xk = xk - fx / dfx0
    print(f"Erro (Modificado): Não convergiu após {max_iter} iterações."); return None, history

def newton_discrete(f, x0, tol, max_iter, h=1e-8):
    """
    Resolve a raiz de uma função f(x) usando o Método de Newton Discreto.
    A derivada é aproximada usando diferenças finitas (f'(x) ≈ (f(x+h) - f(x)) / h).

    Parâmetros:
    f (sympy.Expr): A função simbólica f(x).
    x0 (float): A estimativa inicial para a raiz.
    tol (float): A tolerância para o critério de parada (abs(f(xk)) < tol).
    max_iter (int): O número máximo de iterações.
    h (float, opcional): O passo para a aproximação da derivada. Padrão é 1e-8.

    Retorna:
    tuple: Uma tupla contendo a raiz encontrada (float) ou None se não convergir,
           e uma lista (history) com o histórico de iterações.
    """
    x = sp.symbols('x')
    try: f_num = sp.lambdify(x, f, 'math')
    except Exception as e: print(f"Erro: {e}"); return None, []
    xk, history = x0, []
    for k in range(max_iter):
        fx = f_num(xk); history.append([k, xk, fx])
        if abs(fx) < tol: return xk, history
        try: f_xh = f_num(xk + h); df_discrete = (f_xh - fx) / h
        except ValueError: print("Erro (Discreto): f(x+h)."); return None, history
        if abs(df_discrete) < 1e-15: print(f"Erro (Discreto): Derivada nula (f'({xk:.4f}) \u2248 0)."); return None, history
        xk = xk - fx / df_discrete
    print(f"Erro (Discreto): Não convergiu após {max_iter} iterações."); return None, history

# --- 2. FUNÇÃO PRINCIPAL E INTERFACE COM O USUÁRIO ---

def print_explanations():
    """
    Imprime uma explicação detalhada dos três métodos de Newton
    (Clássico, Modificado e Discreto) que serão comparados pelo script.
    Não possui parâmetros e não retorna valores.
    """
    print("=" * 60); print("      COMPARAÇÃO DOS MÉTODOS DE NEWTON"); print("=" * 60)
    print("\nEste script irá comparar três métodos para encontrar raízes (f(x) = 0):")
    print("\n### 1. Método de Newton Clássico ###"); print("Fórmula: x_{k+1} = x_k - f(x_k) / f'(x_k)")
    print("\n### 2. Método de Newton Modificado ###"); print("Fórmula: x_{k+1} = x_k - f(x_k) / f'(x_0)")
    print("\n### 3. Método de Newton Discreto ###"); print("Fórmula: f'(x_k) \u2248 (f(x_k + h) - f(x_k)) / h")
    print("-" * 60)

# ===================================================================
# FUNÇÃO get_user_input MODIFICADA (AGORA COM PLOTAGEM NÃO-BLOQUEANTE)
# ===================================================================
def get_user_input():
    """
    Obtém a função f(x), suas estimativas iniciais, tolerância e o número máximo
    de iterações do usuário. Também gera e exibe um gráfico da função para auxiliar
    na escolha das estimativas iniciais.

    Parâmetros:
    Nenhum.

    Retorna:
    tuple: Uma tupla contendo:
           - f (sympy.Expr): A função simbólica f(x) inserida pelo usuário.
           - df (sympy.Expr): A derivada simbólica de f(x).
           - x0_list (list): Uma lista de estimativas iniciais (float).
           - tol (float): A tolerância para o critério de parada.
           - max_iter (int): O número máximo de iterações.
           Retorna (None, None, None, None, None) em caso de erro na entrada da função.
    """
    print("\nCONFIGURAÇÃO DO PROBLEMA:")

    # 1. Obter a função
    str_f = input("Digite a função f(x) (ex: x**3 + x**2 - x): ")

    x = sp.symbols('x')
    try:
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
        f_num_plot = sp.lambdify(x, f, 'numpy')

        x_vals = np.linspace(-5, 5, 400)
        y_vals = f_num_plot(x_vals)

        # Ativa o modo interativo
        plt.ion()
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {f}')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Eixo X (y=0)')
        plt.axvline(0, color='gray', linewidth=0.8, linestyle=':')
        plt.title(f'Gráfico da Função: {f}')
        plt.xlabel('x'); plt.ylabel('f(x)')
        plt.grid(True); plt.legend()
        plt.ylim(min(y_vals)-1, max(y_vals)+1)

        # Mostra o gráfico sem bloquear
        plt.show(block=False)
        # Pequena pausa para garantir que a janela seja renderizada
        plt.pause(0.1) # Pausa por 0.1 segundos

        print("\n--- Gráfico exibido em uma janela separada. Deixe-a aberta para consulta! ---")
        print("Use o gráfico para identificar onde a função cruza o eixo X e escolher suas estimativas iniciais!\n")

    except Exception as e:
        print(f"Erro ao gerar o gráfico: {e}")
        print("Continuando sem o gráfico. Por favor, forneça as estimativas iniciais.")

    # 2. Obter estimativa inicial
    print("\n--- O que são as Estimativas Iniciais (x0)? ---")
    print("Para encontrar MÚLTIPLAS raízes, forneça MÚLTIPLAS estimativas.")

    x0_list = []
    while True:
        str_x0s = input("Digite as estimativas iniciais (separadas por vírgula, ex: -2, 0.1, 1): ")
        try:
            x0_list = [float(x.strip()) for x in str_x0s.split(',')]
            if not x0_list: raise ValueError("A lista não pode estar vazia.")
            break
        except ValueError as e:
            print(f"Entrada inválida. Certifique-se de usar números separados por vírgula. Erro: {e}")

    # 3. Obter tolerância
    print("\n--- O que é o Nível de Tolerância (tol)? ---")
    while True:
        try:
            tol = float(input("Digite a tolerância (ex: 1e-7): "))
            if tol <= 0: print("A tolerância deve ser positiva."); else: break
        except ValueError: print("Entrada inválida.")

    # 4. Obter max_iter
    print("\n--- O que é o Número Máximo de Iterações (max_iter)? ---")
    while True:
        try:
            max_iter = int(input("Digite o número máximo de iterações (ex: 100): "))
            if max_iter <= 0: print("O número de iterações deve ser positivo."); else: break
        except ValueError: print("Entrada inválida.")

    return f, df, x0_list, tol, max_iter
# ===================================================================
# FIM DA FUNÇÃO MODIFICADA
# ===================================================================


def main():
    """
    Função principal que orquestra a execução do programa de comparação dos métodos de Newton.
    Chama a função para imprimir as explicações, obtém a entrada do usuário,
    executa os três métodos de Newton para cada estimativa inicial fornecida
    e exibe os resultados em formato de tabela.
    """
    try: sp; tabulate; np; plt
    except NameError:
        print("Erro: Bibliotecas (sympy, tabulate, numpy, matplotlib) não foram encontradas.")
        print("Instale-as com: pip install sympy tabulate numpy matplotlib"); return

    print_explanations()

    f, df, x0_list, tol, max_iter = get_user_input()
    if f is None: return

    print("=" * 60); print("             INICIANDO COMPARAÇÕES"); print("=" * 60)

    for i, x0 in enumerate(x0_list):
        print(f"\n[ TENTATIVA {i + 1} / {len(x0_list)}: Iniciando com x0 = {x0} ]\n")

        summary_data = []

        print(f"--- 1. Clássico (x0 = {x0}) ---")
        root_classic, hist_classic = newton_classic(f, df, x0, tol, max_iter)
        if root_classic is not None: summary_data.append(["Clássico", root_classic, len(hist_classic)-1, hist_classic[-1][2]])
        else: summary_data.append(["Clássico", "Falhou", "-", "-"])

        print(f"--- 2. Modificado (x0 = {x0}) ---")
        root_mod, hist_mod = newton_modified(f, df, x0, tol, max_iter)
        if root_mod is not None: summary_data.append(["Modificado", root_mod, len(hist_mod)-1, hist_mod[-1][2]])
        else: summary_data.append(["Modificado", "Falhou", "-", "-"])

        print(f"--- 3. Discreto (x0 = {x0}) ---")
        root_disc, hist_disc = newton_discrete(f, x0, tol, max_iter, h=1e-8)
        if root_disc is not None: summary_data.append(["Discreto", root_disc, len(hist_disc)-1, hist_disc[-1][2]])
        else: summary_data.append(["Discreto", "Falhou", "-", "-"])

        print("\n" + ("-" * 60))
        print(f"    TABELA DE RESUMO (para x0 = {x0})")
        print(f"    Raízes encontradas (Próximas de {x0}):")
        print(f"-" * 60)
        headers = ["Método", "Raiz Encontrada", "Iterações (k)", "f(raiz)"]
        print(tabulate(summary_data, headers=headers, floatfmt=".10f"))
        print(f"_" * 60)

    print("\nCOMPARAÇÃO CONCLUÍDA.")
    # Fecha todas as janelas de gráfico ao final
    plt.close('all')

if __name__ == "__main__":
    main()
