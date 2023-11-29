import pymc3 as pm
import numpy as np
import pandas as pd


def main():
    data = pd.read_csv('Prices.csv')
    model_regression = pm.Model()
    data['Premium_Binary'] = (data['Premium'] == 'yes').astype(int)

    with pm.Model() as model: #1
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        beta_premium = pm.Normal('beta_premium', mu=0, sigma=10) #pt bonus
        sigma = pm.HalfCauchy('sigma', 5)

        mu = pm.Deterministic('mu',
                              alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive']) + beta_premium * data[
                                  'Premium_Binary'])

        y = pm.Normal('y', mu=mu, sd=sigma, observed=data['Price'])
        trace = pm.sample(5000, tune=1000, cores=2)

    hdi_beta1 = pm.stats.hdi(trace['beta1'], hdi_prob=0.95)
    hdi_beta2 = pm.stats.hdi(trace['beta2'], hdi_prob=0.95)
    hdi_beta_premium = pm.stats.hdi(trace['beta_premium'], hdi_prob=0.95)

    print(f"Estimarea HDI pentru beta1: {hdi_beta1}") #2
    print(f"Estimarea HDI pentru beta2: {hdi_beta2}")
    print(f"Estimarea HDI pentru beta_premium: {hdi_beta_premium}")


    '''
    Ex3. Având în vedere că rezultatle obținute pentru beta1, respectiv pentru beta2 sunt diferite de 0, putem trage concluzia că frecvența procesoruluiâ
    și mărimea hard diskului sunt predictori utili ai prețului de vânzare.Acest lucru înseamnă că acești doi factori au o influență semnificativă 
    asupra determinării prețului de vânzare al PC-urilor.  
    '''

    # Simulare 5000 de extrageri din distribuția predictivă posterioară
    predictive_samples = pm.sample_posterior_predictive(trace, samples=5000, model=model)['y'] #5

    specific_processor_freq = 33
    specific_hard_disk_size = 540

    specific_simulated_prices = pm.sample_posterior_predictive(trace, samples=5000, model=model,
                                                               var_names=['mu'])['mu']

    specific_prices = specific_simulated_prices[:, (data['Speed'] == specific_processor_freq) &
                                                   (np.log(data['HardDrive']) == np.log(specific_hard_disk_size))]

    specific_hdi_price = pm.stats.hdi(specific_prices, hdi_prob=0.90)

    print(f"Estimarea HDI pentru prețul de vânzare așteptat pentru computerul specific: {specific_hdi_price}") #4

    '''
    Bonus. Observăm că intervalul obținut pentru beta_premium este unul vast și poate conține și valoarea 0, acest lucru ar însemna că nu
    întotdeauna atributul Premium este seminficativ in predictia pretului.
    '''

if __name__ == "__main__":
    main()
