from tqdm import tqdm
import numpy as np
import torch
import time
from .buffer import Buffer


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0):      # 普通
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    s = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)
        # action = algo.exploit(state)
        next_state, reward, done, _ = env.step(action)

        mask = False if t == env._max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            continue

        state = next_state



    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer

def collect_demo_two(env, algo, buffer_size, device, std, p_rand, seed=0):      # 普通
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    s = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)
        # action = algo.exploit(state)
        next_state, reward, done, _ = env.step(action)

        mask = False if t == env._max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            print(t)
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0


        state = next_state



    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer

#
# def collect_demo(env, algo, buffer_size, device, std, p_rand, env_fake, seed=0):      # fake airl
#     env.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#
#     buffer = Buffer(
#         buffer_size=buffer_size,
#         state_shape=env.observation_space.shape,
#         action_shape=env.action_space.shape,
#         device=device
#     )
#
#     total_return = 0.0
#     num_episodes = 0
#
#     state = env_fake.reset()
#     env.reset()
#     t = 0
#     episode_return = 0.0
#
#     for _ in tqdm(range(1, buffer_size + 1)):
#         t += 1
#         info = env_fake.infomation()
#
#         action = algo.exploit1(state, info)
#         next_state, reward, done, _ = env_fake.step(action)
#         _, _, done_real, _ = env.step(action)
#         mask = False if t == env._max_episode_steps else done
#         buffer.append(state, action, reward, mask, next_state)
#         episode_return += reward
#
#         if done:
#
#             num_episodes += 1
#             total_return += episode_return
#
#             state = env_fake.reset()
#             t = 0
#             episode_return = 0.0
#
#         if done_real:
#
#             num_episodes += 1
#             total_return += episode_return
#             env.reset()
#             state = env_fake.reset()
#
#             t = 0
#             episode_return = 0.0
#
#
#         state = next_state
#
#
#     print(f'Mean return of the expert is {total_return / num_episodes}')
#     return buffer


# def collect_demo(env, algo, buffer_size, device, std, p_rand, env_fake, seed=0):      # fake daairl
#     env.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#
#     buffer = Buffer(
#         buffer_size=buffer_size,
#         state_shape=env.observation_space.shape,
#         action_shape=env.action_space.shape,
#         device=device
#     )
#
#     total_return = 0.0
#     num_episodes = 0
#
#     state = env_fake.reset()
#     state_real = env.reset()
#     state_real[3] = state[3]
#     state_real[4] = state[4]
#     state_real[5] = state[5]
#     t = 0
#     episode_return = 0.0
#
#     for _ in tqdm(range(1, buffer_size + 1)):
#         t += 1
#
#         info = env.infomation()
#         # action = algo.exploit(state)
#         action = algo.exploit1(state_real, info)
#         next_state_real, reward_real, done_real, _ = env.step(action)
#         next_state, reward, done, _ = env_fake.step(action)
#
#         next_state_real[3] = next_state[3]
#         next_state_real[4] = next_state[4]
#         next_state_real[5] = next_state[5]
#
#
#
#         mask = False if t == env._max_episode_steps else done
#         buffer.append(state_real, action, reward, mask, next_state_real)
#         episode_return += reward
#         # if not info:
#         #     file_force_y = open('D:\\force_sensor\\action_x_real.txt', mode='a')
#         #     file_force_x = open('D:\\force_sensor\\action_y_real.txt', mode='a')
#         #     file_force_z = open('D:\\force_sensor\\action_z_real.txt', mode='a')
#         #     file_force_xx = open('D:\\force_sensor\\force_x_real.txt', mode='a')
#         #     # file_depth_z = open('D:\\force_sensor\\depth_z.txt', mode='a')
#         #     # file_depth_z.write(f"{t}\n")
#         #     file_force_x.write(f"{action[2]}\n")
#         #     file_force_y.write(f"{action[1]}\n")
#         #     file_force_z.write(f"{action[0]}\n")
#         #     file_force_xx.write(f"{state_real[3]}\n")
#         #     file_force_y.close()
#         #     file_force_x.close()
#         #     file_force_z.close()
#         #     file_force_xx.close()
#         # if done:
#         #
#         #     num_episodes += 1
#         #     total_return += episode_return
#         #     state = env_fake.reset()
#         #     env.reset()
#         #     t = 0
#         #     episode_return = 0.0
#
#         if done_real:
#             num_episodes += 1
#             total_return += episode_return
#             state_real = env.reset()
#             state = env_fake.reset()
#             t = 0
#             episode_return = 0.0
#
#         state_real = next_state_real
#         state = next_state
#
#
#
#     print(f'Mean return of the expert is {total_return / num_episodes}')
#     return buffer
#
def collect_demo1(env, algo, buffer_size, device, std, p_rand, seed=0):      # Daairl
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer0 = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    buffer1 = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    r = 0
    s = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1), disable=True):
        t += 1

        info = env.infomation()


        if np.random.rand() < p_rand:
            action = env.action_space.sample()
            change = env.change_space.sample()
        else:
            action, change = algo.exploit1(state, info)
            action = add_random_noise(action, std)
        next_state, reward, done, _ = env.step(action, change)


            # file_depth_z.close()




        mask = False if t == env._max_episode_steps else done
        # if info:
        #     buffer0.append(state, action, reward, mask, next_state)
        # else:
        #     buffer1.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return

            state = env.reset()
            t = 0
            s += 1
            episode_return = 0.0

        else:
            state = next_state

        if s == 100:
            break

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer0, buffer1

def collect_demo_DEITC(env, realenv, algo, buffer_size, device, std, p_rand, seed=0):      # Daairl
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer0 = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    buffer1 = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    position = env.posi()
    realenv.reset([position[0][0], position[1][0], position[2][0]])
    t = 0
    r = 0
    s = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1), disable=True):
        t += 1

        info = env.infomation()

        action, change = algo.exploit1(state, info)
        # action = add_random_noise(action, std)
        next_state, reward, done, _ = env.step(action, change)
        chang = env.gripper()
        realenv.step(action, chang)

            # file_depth_z.close()




        mask = False if t == env._max_episode_steps else done
        # if info:
        #     buffer0.append(state, action, reward, mask, next_state)
        # else:
        #     buffer1.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return

            state = env.reset()
            position = env.posi()
            realenv.reset([position[0][0], position[1][0], position[2][0]])
            t = 0
            s += 1
            episode_return = 0.0

        else:
            state = next_state

        if s == 1:
            break

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer0, buffer1

def collect_demon(env, algo, buffer_size, device, std, p_rand, seed=0):      # Daairl
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer0 = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    buffer1 = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    r = 0
    s = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        info = env.infomation()


        action = algo.exploit1(state, info)
        # action = add_random_noise(action, std)
        next_state, reward, done, _ = env.step(action)
        time.sleep(1/120)


            # file_depth_z.close()




        mask = False if t == env._max_episode_steps else done
        # if info:
        #     buffer0.append(state, action, reward, mask, next_state)
        # else:
        #     buffer1.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return

            state = env.reset()
            t = 0
            s += 1
            episode_return = 0.0

        else:
            state = next_state
        if s == 100:
            break


    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer0, buffer1

def display(env, algo, buffer_size, device, std, p_rand, seed=0):      # Daairl
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    r = 0
    s = 0
    episode_return = 0.0
    info = 0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        info = env.infomation()

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit_SC(state, info)
            action = add_random_noise(action, std)

        # action = algo.exploit_panda_two(state, info)
        # action = add_random_noise(action, std)

        next_state, reward, done, info = env.step(action)
        time.sleep(1/120)


        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return

            state = env.reset()
            t = 0
            s += 1
            episode_return = 0.0

        else:
            state = next_state
        if s == 50:
            break


    print(f'Mean return of the expert is {total_return / num_episodes}')
    return 0
# def collect_demo1(env, env_fake, algo, buffer_size, device, std, p_rand, seed=0):      # Daairl
#     env.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#
#     buffer = Buffer(
#         buffer_size=buffer_size,
#         state_shape=env.observation_space.shape,
#         action_shape=env.action_space.shape,
#         device=device
#     )
#
#     total_return = 0.0
#     num_episodes = 0
#
#     state = env.reset()
#     env_fake.reset()
#     t = 0
#     r = 0
#     s = 0
#     episode_return = 0.0
#
#     for _ in tqdm(range(1, buffer_size + 1)):
#         t += 1
#
#         info = env.infomation()
#
#         if not info and r == 0:
#             time.sleep(1)
#             r = 1
#         action = algo.exploit1(state, info)
#         next_state, reward, done, _ = env.step(action)
#         env_fake.step(action)
#         # if not info and s == 1:
#         #     file_force_y = open('D:\\force_sensor\\force_y.txt', mode='a')
#         #     file_force_x = open('D:\\force_sensor\\force_x.txt', mode='a')
#         #     file_force_z = open('D:\\force_sensor\\force_z.txt', mode='a')
#         #     file_depth_z = open('D:\\force_sensor\\depth_z.txt', mode='a')
#         #     file_depth_z.write(f"{t}\n")
#         #     file_force_x.write(f"{state[5]}\n")
#         #     file_force_y.write(f"{state[3]}\n")
#         #     file_force_z.write(f"{state[4]}\n")
#         #     file_force_y.close()
#         #     file_force_x.close()
#         #     file_force_z.close()
#         #     file_depth_z.close()
#         # time.sleep(1/240)
#
#
#
#         mask = False if t == env._max_episode_steps else done
#         buffer.append(state, action, reward, mask, next_state)
#         episode_return += reward
#
#         if done:
#             s += 1
#             if s == 101:
#                 break
#             num_episodes += 1
#             total_return += episode_return
#
#             state = env.reset()
#             t = 0
#             s += 1
#             episode_return = 0.0
#
#
#         state = next_state
#
#
#
#     print(f'Mean return of the expert is {total_return / num_episodes}')
#     return buffer