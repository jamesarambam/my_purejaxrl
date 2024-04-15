from purejaxrl.my_environment.safetyGrid import SafeGridworld
from purejaxrl.my_environment.safetyGrid import EnvParams
from pdb import set_trace
import gymnax
import jax


key = jax.random.PRNGKey(0)

env1 = gymnax.make('CartPole-v1')

# env = SafeGridworld()


env = SafeGridworld()
set_trace()
env.step_env(key, 1, 0, env_params)


exit()


tran_prob = 0.99
s = 6
p = jax.random.uniform(key)

def true_fn(s):
    return s + 1

def false_fn(s):
    return s - 1

ns = jax.lax.cond(p < tran_prob, true_fn(s), false_fn(s))

print(p)
print(ns)

#set_trace()