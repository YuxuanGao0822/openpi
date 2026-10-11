import argparse

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.t2mem.policy import T2MemPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-writes", action="store_true")
    parser.add_argument("--denoise-steps", type=int, default=10)
    args = parser.parse_args()
    policy = T2MemPolicy(
        args.checkpoint, seed=args.seed, disable_writes=args.disable_writes, num_steps=args.denoise_steps
    )
    config = policy.model.config
    WebsocketPolicyServer(
        policy,
        port=args.port,
        metadata={
            "method": "pi05_baseline" if policy.model.stage == "stage1" else "T2Mem",
            "horizon": config.horizon,
            "execute": config.execute,
            "write_stride": config.write_stride,
            "demo_write_offset": config.demo_write_offset,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
