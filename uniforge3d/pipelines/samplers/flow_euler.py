"""Shared flow-matching Euler and RK4 sampler for every cascade stage."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm


# Sampler configuration option: choose between "euler" and "rk4"
DEFAULT_SAMPLER_METHOD: Literal["euler", "rk4"] = "rk4"


Prediction = Callable[[Any, float, bool], Any]


def model_predictor(
    model: torch.nn.Module,
    cond: Any,
    neg_cond: Any,
    *,
    timestep_scale: float = 1000.0,
    **model_kwargs,
) -> Prediction:
    """Adapt a standard conditioned flow model to the shared sampler."""

    def predict(sample, time: float, conditioned: bool):
        timestep = torch.full(
            (sample.shape[0],),
            timestep_scale * time,
            device=sample.device,
            dtype=torch.float32,
        )
        condition = cond if conditioned else neg_cond
        return model(sample, timestep, condition, **model_kwargs)

    return predict


class FlowEulerSampler:
    """Euler / RK4 CFG sampling shared by sparse structure, shape, and texture."""

    def __init__(self, sigma_min: float):
        self.sigma_min = sigma_min

    def _prediction_to_xstart(self, x_t, t: float, prediction):
        return (
            (1 - self.sigma_min) * x_t
            - (self.sigma_min + (1 - self.sigma_min) * t) * prediction
        )

    def _xstart_to_prediction(self, x_t, t: float, x_start):
        return (
            (1 - self.sigma_min) * x_t - x_start
        ) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _guided_prediction(
        self,
        predict: Prediction,
        sample,
        t: float,
        guidance_strength: float,
        guidance_interval: tuple[float, float],
        guidance_rescale: float,
        guidance_as_difference: bool,
    ):
        strength = (
            guidance_strength
            if guidance_interval[0] <= t <= guidance_interval[1]
            else 1.0
        )
        if strength == 1:
            return predict(sample, t, True)
        if strength == 0:
            return predict(sample, t, False)

        positive = predict(sample, t, True)
        negative = predict(sample, t, False)
        if guidance_as_difference:
            guided = negative + strength * (positive - negative)
        else:
            guided = strength * positive + (1 - strength) * negative
        if guidance_rescale <= 0:
            return guided

        x0_positive = self._prediction_to_xstart(sample, t, positive)
        x0_guided = self._prediction_to_xstart(sample, t, guided)
        dimensions = list(range(1, x0_guided.ndim))
        positive_std = x0_positive.float().std(dim=dimensions, keepdim=True)
        guided_std = x0_guided.float().std(dim=dimensions, keepdim=True).clamp_min(1e-6)
        scale = (positive_std / guided_std).to(x0_guided.dtype)
        x0_rescaled = x0_guided * scale
        x0 = guidance_rescale * x0_rescaled + (1 - guidance_rescale) * x0_guided
        return self._xstart_to_prediction(sample, t, x0)

    @torch.no_grad()
    def sample(
        self,
        noise,
        predict: Prediction,
        *,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: tuple[float, float] = (0.0, 1.0),
        guidance_rescale: float = 0.0,
        guidance_as_difference: bool = False,
        verbose: bool = True,
        tqdm_desc: str = "Sampling ",
        sampler_method: Literal["euler", "rk4"] = DEFAULT_SAMPLER_METHOD,
    ):
        """Run the common Euler or RK4 CFG update using a stage-specific predictor."""
        sample = noise
        times = np.linspace(1, 0, steps + 1)
        times = rescale_t * times / (1 + (rescale_t - 1) * times)
        time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))
        desc = f"{tqdm_desc} ({sampler_method.upper()})"
        for current, previous in tqdm(
            time_pairs,
            desc=desc,
            disable=not verbose,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
        ):
            dt = current - previous

            if sampler_method == "euler":
                prediction = self._guided_prediction(
                    predict,
                    sample,
                    current,
                    guidance_strength,
                    guidance_interval,
                    guidance_rescale,
                    guidance_as_difference,
                )
                sample = sample - dt * prediction

            elif sampler_method == "rk4":
                k1 = self._guided_prediction(
                    predict,
                    sample,
                    current,
                    guidance_strength,
                    guidance_interval,
                    guidance_rescale,
                    guidance_as_difference,
                )
                
                sample_k2 = sample - 0.5 * dt * k1
                k2 = self._guided_prediction(
                    predict,
                    sample_k2,
                    current - 0.5 * dt,
                    guidance_strength,
                    guidance_interval,
                    guidance_rescale,
                    guidance_as_difference,
                )
                
                sample_k3 = sample - 0.5 * dt * k2
                k3 = self._guided_prediction(
                    predict,
                    sample_k3,
                    current - 0.5 * dt,
                    guidance_strength,
                    guidance_interval,
                    guidance_rescale,
                    guidance_as_difference,
                )
                
                sample_k4 = sample - dt * k3
                k4 = self._guided_prediction(
                    predict,
                    sample_k4,
                    previous,
                    guidance_strength,
                    guidance_interval,
                    guidance_rescale,
                    guidance_as_difference,
                )
                
                sample = sample - (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown sampling method: {sampler_method}. Choose 'euler' or 'rk4'.")

        return edict(samples=sample)