import sympy

y_hat = []
y = [1, 2, 3, 4]
x = [3, 4, 5, 6]
w, b = sympy.symbols('w b')

def linearFunction(x):
    return w * x + b

def linearRegression(**kwargs):
    global y_hat
    y_hat = []
    for i in range(len(x)):
        expr = linearFunction(x[i])
        if(kwargs.get('w') is not None):
            expr = expr.subs({w: kwargs.get('w')})
        if(kwargs.get('b') is not None):
            expr = expr.subs({b: kwargs.get('b')})
        if((kwargs.get('w')) is not None and (kwargs.get('b') is not None)):
            expr = expr.subs({w: kwargs.get('w'), b: kwargs.get('b')})
        y_hat.append(expr)

def meanSquaredError(y_hat, y):
    M = len(y)
    SE = 0
    for i in range(M):
       SE += (y_hat[i] - y[i])**2
    return SE/(2*M)

def gradientDescent(learningRate = 0.05, n_iter = 10000):
    g_w, g_b = 10, 10
    previousDW,previousDB = float('inf'),float('inf')
    
    for i in range(0,n_iter):
        linearRegression()
        MSE = meanSquaredError(y_hat, y)
        dw = sympy.diff(MSE, w).evalf(subs={w: g_w, b: g_b})
        db = sympy.diff(MSE, b).evalf(subs={w: g_w, b: g_b})
        
        if((abs(dw) < previousDW) and (abs(db) < previousDB)):
            temp_w = g_w-learningRate * dw
            temp_b = g_b-learningRate * db
            
            g_w,g_b = temp_w,temp_b
        
        if ((abs(dw) <= 1e-6) and (abs(db) <= 1e-6)):
            print("Local Minimum reached at iteration:", i)
            break
    
    print("w: {}, b: {}".format(g_w, g_b))
    
    return {
        w: g_w,
        b: g_b
    }

tuned_hps = gradientDescent()
linearRegression(w=tuned_hps.get(w), b=tuned_hps.get(b))

print("y_hat: ", y_hat)
print("MSE: ", meanSquaredError(y_hat, y))
