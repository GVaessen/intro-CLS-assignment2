import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def XYZ(t, y0, beta, gamma, mu, v):
    '''
    Stochastic SIR equations with process noise
    '''
    X = y0[0]
    Y = y0[1]
    Z = y0[2]
    N = X+Y+Z

    dX_dt = v*N - (beta*X*Y/N + N*0.1*np.random.normal()) - mu*X
    dY_dt = (beta*X*Y/N + N*0.1*np.random.normal()) - gamma*Y - mu*Y
    dZ_dt = gamma*Y - mu*Z

    return dX_dt, dY_dt, dZ_dt


def phase_diagram(beta, gamma, start=0, end=50, S0 = np.linspace(0, 1, 16), dt=0.1,
I0 = np.linspace(0, 1, 16)[::-1], R0 = [0 for i in  range(16)], immunity = True,
ax=None, mu=0, v=0, threshold=True, show_R_0=True,
method='GDA', **plt_kwargs):
    '''
    Creates a phase plot with the infected proportion on the x-axis and the
    susceptible proportion on the y-axis

    beta, gamma, mu, v : floats, parameters for the SIR model
    start, end : numbers, start time and end time inclusive, default is 0, 50
    dt: number, timestep at which the numerical solver operates, default is 0.1
    S0, I0, R0: array like, arrray of initial values for S, I and R must be the
    same length, default R(0)=0 and I(0), S(0) in [0, 1]
    immunity: Boolean if True plots lines for R(0) > 0, default is True
    ax: ax1 or ax2 etc. for specifying subplots, default is None
    threshold: Boolean, if True shows the epidemic treshold if withn xlim, default is True
    show_R_0: Boolean, if True shows the value of R_0 in the upper right corner, default is True
    mehod: which method to use, options are deterministic SIR, stochastic SIR, GDA
    '''

    if ax is None:
        ax = plt.gca()

    R_0 = beta / (gamma+mu)

    # plot initial conditions
    ax.scatter(S0, I0, color='black', s=15)

    # lines for R(0) = 0
    for s, i, r in zip(S0, I0, R0):
        # if method == 'GDA':


        if method == 'stochastic SIR':
            sol = solve_ivp(XYZ, (start, end), y0=[s, i, r], args=(beta, gamma, mu, v),
            t_eval=np.arange(start, end+dt, dt))
        else:
            sol = solve_ivp(SIR, (start, end), y0=[s, i, r], args=(beta, gamma, mu, v),
            t_eval=np.arange(start, end+dt, dt))
        ax.plot(sol['y'][0], sol['y'][1], color='black')

    # lines for R(0) > 0
    if immunity:
        s0 = np.arange(0.5, 1, 0.1)
        s0 = np.append(s0, 0.95)
        i0 = np.array([0.001 for s in s0])
        c0 = np.array([0 for s in s0])
        r0 = np.array([1-(s+i+c) for s, i, c in zip(s0, i0, c0)])

        # plot initials condition
        ax.scatter(s0, i0, color='black', s=15)

        for s, i, r in zip(s0, i0, r0):
            if method == 'stochastic SIR':
                sol = solve_ivp(XYZ, (start, end), y0=[s, i, r], args=(beta, gamma, mu, v),
                t_eval=np.arange(start, end+dt, dt))
            else:
                sol = solve_ivp(SIR, (start, end), y0=[s, i, r], args=(beta, gamma, mu, v),
                t_eval=np.arange(start, end+dt, dt))
            ax.plot(sol['y'][0], sol['y'][1], color='black')

    # line where S+I=1
    ax.plot(np.linspace(0, 1), np.linspace(0, 1)[::-1], color='grey')

    # show value of R_0
    if show_R_0:
        anchored_text = AnchoredText(rf'$R_0 = {round(R_0,2)}$', loc=1)
        ax.add_artist(anchored_text)

    # show epidemic threshold
    if R_0 > 1:
        if threshold:
            ax.vlines(1/R_0, 0, -1/R_0+1, linestyle='--', colors='grey')
            ax.text(1/R_0*0.95, (-1/R_0+1)*1.03, r'$1 / R_0$')

    ax.set_xlabel('Susceptibles')
    ax.set_ylabel('Infected')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def solve_SIR(S0, I0, R0, beta, gamma, mu=0, v=0, start=0, end=50, dt=0.1):
    sol = solve_ivp(XYZ, (start, end), y0=[S0, I0, R0], args=(beta, gamma, mu, v),
    t_eval=np.arange(start, end+dt, dt))
    return sol['t'], sol['y'][0], sol['y'][1], sol['y'][2]


def time_plot(X0, Y0, Z0, beta, gamma, mu=0, v=0,
 steps=100, scale=False, ax=None, **plt_kwargs):

    if ax is None:
        ax = plt.gca()

    X, Y, Z, t = GDA(X0, Y0, Z0, rates=rates, steps=steps, scale=scale)

    ax.plot(t, X, label='Susceptible', color='black')
    ax.plot(t, Y, '--', label='Infected', color='black')
    ax.plot(t, Z, '-.', label='Recovered', color='grey')

    ax.set_xlabel('time')
    ax.set_xlim(0, max(t))
    ax.set_ylim(0, max([max(X), max(Y), max(Z)]))
    ax.set_ylabel('# individuals')
    ax.legend(loc='upper right')
    return ax


def SIR(t, y0, beta, gamma, mu=0, v=0):
    '''
    Deterministic SIR equations with demography
    '''

    S = y0[0]
    I = y0[1]
    R = y0[2]

    dS_dt = v - beta*S*I - mu*S
    dI_dt = beta*S*I - gamma*I - mu*I
    dR_dt = gamma*I - mu*R
    return dS_dt, dI_dt, dR_dt


def SIR_rates(X, Y, Z, beta, gamma, mu=0, v=0):
    '''
    Deterministic SIR equations with demography
    '''
    N = X+Y+Z
    dX_dt = v - beta*X*Y - mu*X
    dY_dt = beta*X*Y - gamma*Y - mu*Y
    dZ_dt = gamma*Y - mu*Z
    return {'birth': mu*N, 'transmission': beta*X*Y/N, 'recovery': gamma*Y,
    'death X': mu*X, 'death Y': mu*Y, 'death Z':mu*Z}


def GDA(X0, Y0, Z0, beta, gamma, mu=0, v=0, steps=100, scale=False):
    '''
    Gillespie's Direct Algorithm
    '''
    # total population
    N = [X0 + Y0 + Z0]
    #  initial values
    X = [X0]
    Y = [Y0]
    Z = [Z0]

    time = [0]

    for i in range(steps):

        SIR_rates(X[-1], Y[-1], Z[-1], beta, gamma, mu=mu, v=v)
        # inf_rate =

        R_total = sum(rates.values())
        dt = -1/R_total * np.log(np.random.random())
        P = np.random.random() * R_total

        Rm = [0]

        # changes in population
        dX = 0
        dY = 0
        dZ = 0
        for event, rate in rates.items():
            if sum(Rm) < P <= sum(rates.values()):
                if event == 'birth':
                    dX += 1
                elif event == 'infection':
                    dY += 1
                elif event == 'recovery':
                    dZ += 1
                else:
                    if X[-1] > np.random.random():
                        dX -= 1
                    elif Y[-1] > np.random.random():
                        dY -= 1
                    else:
                        dZ -= 1
            Rm += [rate]
        X += [X[-1] + dX]
        Y += [Y[-1] + dY]
        Z += [Z[-1] + dZ]
        N += [X[-1] + Y[-1] + Z[-1]]

        time += [sum(time)+dt]

    # returns XYZ as fractions
    if scale:
        X = np.array(X)/np.array(N)
        Y = np.array(Y)/np.array(N)
        Z = np.array(Z)/np.array(N)

    return X, Y, Z, time

time_plot(1000, 10e4*2.5e-4, 10e4-(1000+2.5e-4), rates={'infection':520, 'recovery':365.25, 'birth':1/70, 'death':1/70})

plt.ylim(100, 140)
plt.show()
