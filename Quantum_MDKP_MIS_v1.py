# Quantum_MDKP_MIS
# MDKP - Multi-Dimensional Knapsack
# MIS - Maximum Independent Set



import csv
import hashlib
import itertools
import math
import random
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.special import logsumexp

random.seed(39)
np.random.seed(39)

FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_2980.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_plus_1736.csv"),
)

N = 39
K = 7
QUBITS = 6
DIM = 64
PARAMETERS_PER_REGISTER = 12

WINDOWS = (1, 3)
REGULARIZATIONS = (0.01, 0.10)

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

VQE_MAX_ITERATIONS = 80
VQE_GRADIENT_TOLERANCE = 1e-6
CONFLICT_PENALTY = 2.0

STATE_INDICES = np.arange(DIM)
NUMBER_INDICES = np.arange(N)

VALID = np.array(
    [
        (NUMBER_INDICES >= slot)
        & (NUMBER_INDICES <= N - K + slot)
        for slot in range(K)
    ],
    dtype=bool,
)

INCREASING = (
    NUMBER_INDICES[:, None] < NUMBER_INDICES[None, :]
)

LEGAL_PAIRS = np.where(
    INCREASING[None, :, :],
    np.zeros((K - 1, N, N)),
    -np.inf,
)

CONFLICT = np.zeros((DIM, DIM))
CONFLICT[:N, :N] = (~INCREASING).astype(float)

ZERO_INDICES = tuple(
    STATE_INDICES[(STATE_INDICES & (1 << qubit)) == 0]
    for qubit in range(QUBITS)
)

ONE_INDICES = tuple(
    indices + (1 << qubit)
    for qubit, indices in enumerate(ZERO_INDICES)
)

CX_PERMUTATIONS = tuple(
    np.where(
        (STATE_INDICES & (1 << control)) != 0,
        STATE_INDICES ^ (1 << (control + 1)),
        STATE_INDICES,
    )
    for control in range(QUBITS - 1)
)


class DistributionError(ArithmeticError):
    pass


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    reader = csv.reader(
        raw.decode("utf-8-sig").splitlines()
    )

    for line_number, row in enumerate(reader, start=1):
        require(
            len(row) == K,
            f"{path.name}: red {line_number} nema sedam kolona.",
        )

        try:
            values = tuple(int(value.strip()) for value in row)
        except ValueError as error:
            raise ValueError(
                f"{path.name}: necelobrojan red {line_number}."
            ) from error

        require(
            len(set(values)) == K
            and min(values) >= 1
            and max(values) <= N,
            f"{path.name}: nelegalan red {line_number}.",
        )

        require(
            values == tuple(sorted(values)),
            f"{path.name}: red {line_number} nije rastuci.",
        )

        rows.append(values)

    require(
        len(rows) >= 5,
        f"{path.name}: potrebno je najmanje pet kola "
        "za zadatu podelu na trening, validaciju i test.",
    )

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def input_states(data, times, window):
    times = np.asarray(times, dtype=np.int64)

    require(
        window in WINDOWS
        and times.ndim == 1
        and len(times) > 0
        and np.all((times >= 1) & (times <= len(data))),
        "Neispravan prozor ili vreme ulaznog uzorka.",
    )

    base = np.zeros((len(times), DIM), dtype=float)
    base[:, 0] = 1.0

    for lag in range(1, window + 1):
        source = times - lag
        available = source >= 0
        start = 1 + 8 * (lag - 1)

        base[available, start:start + K] = (
            data[source[available]].astype(float)
            + 1.0
            - 20.0
        ) / (19.0 * math.sqrt(window))

        base[available, start + K] = 1.0 / math.sqrt(window)

    result = np.empty(
        (len(times), K, DIM),
        dtype=np.complex128,
    )

    for slot in range(K):
        values = base.copy()
        values[:, 63] = (slot + 1.0) / K

        phase = np.exp(
            1j * (slot + 1.0) * STATE_INDICES / DIM
        )

        values = values.astype(np.complex128) * phase[None, :]
        norms = np.linalg.norm(values, axis=1)

        require(
            np.all(np.isfinite(norms)) and np.all(norms > 0),
            "Neispravna norma ulaznog stanja.",
        )

        result[:, slot] = values / norms[:, None]

    return result


def fit_unitaries(data, transition_count, window, regularization):
    require(
        1 <= transition_count < len(data),
        "Neispravan broj trening prelaza.",
    )

    times = np.arange(1, transition_count + 1)
    states = input_states(data, times, window)
    targets = data[times]

    unitaries = np.empty(
        (K, DIM, DIM),
        dtype=np.complex128,
    )

    identity = np.eye(DIM, dtype=np.complex128)

    for slot in range(K):
        cross = np.zeros((DIM, DIM), dtype=np.complex128)

        # Srednja vrednost |target><input|.
        np.add.at(
            cross,
            targets[:, slot],
            np.conjugate(states[:, slot]),
        )

        cross /= transition_count

        # Regularizacija unitarnog operatora prema identitetu.
        cross += regularization * identity

        left, _, right_adjoint = np.linalg.svd(
            cross,
            full_matrices=True,
        )
        operator = left @ right_adjoint

        require(
            np.all(np.isfinite(operator))
            and np.allclose(
                operator.conj().T @ operator,
                identity,
                atol=2e-10,
                rtol=0.0,
            ),
            "Nauceni operator nije unitaran.",
        )

        unitaries[slot] = operator

    return unitaries


def forecast_amplitudes(unitaries, states):
    amplitudes = np.einsum(
        "jab,jb->ja",
        unitaries,
        states,
        optimize=True,
    )

    probabilities(amplitudes)
    return amplitudes


def probabilities(amplitudes):
    result = np.abs(amplitudes) ** 2

    require(
        np.all(np.isfinite(result))
        and np.allclose(
            result.sum(axis=-1),
            1.0,
            atol=2e-10,
            rtol=0.0,
        ),
        "Kvantno stanje nije normalizovano.",
    )

    return result


def rotate_y(values, angle, qubit):
    zero = ZERO_INDICES[qubit]
    one = ONE_INDICES[qubit]

    cosine = math.cos(angle / 2.0)
    sine = math.sin(angle / 2.0)

    result = values.copy()
    result[zero] = cosine * values[zero] - sine * values[one]
    result[one] = sine * values[zero] + cosine * values[one]

    return result


def circuit_forward(state, parameters, record=False):
    values = state.copy()
    cache = []

    for qubit in range(QUBITS):
        if record:
            cache.append(("ry", qubit, qubit, values))

        values = rotate_y(
            values,
            float(parameters[qubit]),
            qubit,
        )

    for control in range(QUBITS - 1):
        if record:
            cache.append(("cx", control, None, None))

        values = values[CX_PERMUTATIONS[control]]

    for qubit in range(QUBITS):
        index = QUBITS + qubit

        if record:
            cache.append(("ry", qubit, index, values))

        values = rotate_y(
            values,
            float(parameters[index]),
            qubit,
        )

    return values, cache


def circuit_backward(adjoint, parameters, cache):
    gradient = np.zeros(PARAMETERS_PER_REGISTER)
    values = adjoint

    for gate, qubit, index, before in reversed(cache):
        if gate == "cx":
            values = values[CX_PERMUTATIONS[qubit]]
            continue

        angle = float(parameters[index])
        cosine = math.cos(angle / 2.0)
        sine = math.sin(angle / 2.0)

        zero = ZERO_INDICES[qubit]
        one = ONE_INDICES[qubit]

        derivative_zero = -0.5 * (
            sine * before[zero] + cosine * before[one]
        )
        derivative_one = 0.5 * (
            cosine * before[zero] - sine * before[one]
        )

        gradient[index] = float(np.real(
            np.vdot(values[zero], derivative_zero)
            + np.vdot(values[one], derivative_one)
        ))

        values = rotate_y(values, -angle, qubit)

    return gradient


def build_hamiltonian(item_values):
    require(
        item_values.shape == (K, DIM)
        and np.all(np.isfinite(item_values))
        and np.all(item_values >= 0.0),
        "Neispravne vrednosti MDKP predmeta.",
    )

    # Ukupna nagrada podeljena sa K nalazi se u [0, 1].
    # Svaka povreda dodaje kaznu 2.
    local_energy = np.full((K, DIM), CONFLICT_PENALTY)

    local_energy[:, :N] = (
        -item_values[:, :N] / K
        + CONFLICT_PENALTY * (~VALID)
    )

    return local_energy


def vqe_objective(flat_parameters, initial_states, local_energy):
    parameters = flat_parameters.reshape(
        K,
        PARAMETERS_PER_REGISTER,
    )

    amplitudes = np.empty((K, DIM), dtype=np.complex128)
    caches = []

    for slot in range(K):
        amplitudes[slot], cache = circuit_forward(
            initial_states[slot],
            parameters[slot],
            record=True,
        )
        caches.append(cache)

    born = probabilities(amplitudes)

    energy = float(np.sum(born * local_energy))
    derivative = local_energy.copy()

    for slot in range(K - 1):
        right_effect = CONFLICT @ born[slot + 1]
        left_effect = born[slot] @ CONFLICT

        energy += CONFLICT_PENALTY * float(
            np.dot(born[slot], right_effect)
        )

        derivative[slot] += CONFLICT_PENALTY * right_effect
        derivative[slot + 1] += CONFLICT_PENALTY * left_effect

    gradient = np.empty_like(parameters)

    for slot in range(K):
        adjoint = 2.0 * derivative[slot] * amplitudes[slot]

        gradient[slot] = circuit_backward(
            adjoint,
            parameters[slot],
            caches[slot],
        )

    require(
        np.isfinite(energy) and np.all(np.isfinite(gradient)),
        "Nekonacna VQE energija ili gradijent.",
    )

    return energy, gradient.ravel()


def final_vqe_amplitudes(initial_states, flat_parameters):
    parameters = flat_parameters.reshape(
        K,
        PARAMETERS_PER_REGISTER,
    )

    amplitudes = np.empty_like(initial_states)

    for slot in range(K):
        amplitudes[slot], _ = circuit_forward(
            initial_states[slot],
            parameters[slot],
        )

    probabilities(amplitudes)
    return amplitudes


def solve_vqe(initial_states):
    item_values = probabilities(initial_states)
    local_energy = build_hamiltonian(item_values)

    indices = np.arange(
        K * PARAMETERS_PER_REGISTER,
        dtype=float,
    )
    initial_parameters = 0.15 * np.sin(indices + 1.0)

    initial_energy, _ = vqe_objective(
        initial_parameters,
        initial_states,
        local_energy,
    )

    best_parameters = initial_parameters.copy()
    best_energy = initial_energy

    def tracked_objective(parameters):
        nonlocal best_parameters, best_energy

        energy, gradient = vqe_objective(
            parameters,
            initial_states,
            local_energy,
        )

        if energy < best_energy:
            best_energy = energy
            best_parameters = parameters.copy()

        return energy, gradient

    result = minimize(
        tracked_objective,
        initial_parameters,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": VQE_MAX_ITERATIONS,
            "gtol": VQE_GRADIENT_TOLERANCE,
            "ftol": 1e-11,
            "maxls": 30,
        },
    )

    amplitudes = final_vqe_amplitudes(
        initial_states,
        best_parameters,
    )

    _, gradient = vqe_objective(
        best_parameters,
        initial_states,
        local_energy,
    )

    return {
        "amplitudes": amplitudes,
        "parameters": best_parameters,
        "energy": best_energy,
        "initial_energy": initial_energy,
        "gradient_norm": float(np.max(np.abs(gradient))),
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "item_values": item_values,
    }


def legal_log_unary(amplitudes):
    magnitudes = np.abs(amplitudes[:, :N])
    unary = np.full((K, N), -np.inf)

    np.log(
        magnitudes,
        out=unary,
        where=magnitudes > 0.0,
    )
    unary *= 2.0

    return np.where(VALID, unary, -np.inf)


def legal_distribution(unary):
    slots, states = unary.shape

    increasing = (
        np.arange(states)[:, None] < np.arange(states)[None, :]
    )

    forward = np.full_like(unary, -np.inf)
    backward = np.full_like(unary, -np.inf)

    forward[0] = unary[0]

    values = unary[0].copy()
    parents = np.zeros((slots, states), dtype=np.int64)

    for slot in range(1, slots):
        forward_terms = np.where(
            increasing,
            forward[slot - 1, :, None],
            -np.inf,
        )

        forward[slot] = (
            unary[slot] + logsumexp(forward_terms, axis=0)
        )

        candidates = np.where(
            increasing,
            values[:, None],
            -np.inf,
        )

        parents[slot] = np.argmax(candidates, axis=0)

        values = (
            unary[slot]
            + candidates[parents[slot], np.arange(states)]
        )

    log_partition = float(logsumexp(forward[-1]))

    if not np.isfinite(log_partition):
        raise DistributionError(
            "Zavrsno kvantno stanje nema pozitivnu legalnu masu."
        )

    last = int(np.argmax(values))

    if not np.isfinite(values[last]):
        raise DistributionError(
            "Ne postoji legalan maksimum."
        )

    path = np.zeros(slots, dtype=np.int64)
    path[-1] = last

    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parents[slot, path[slot]]

    backward[-1] = 0.0

    for slot in range(slots - 2, -1, -1):
        terms = np.where(
            increasing,
            (
                unary[slot + 1]
                + backward[slot + 1]
            )[None, :],
            -np.inf,
        )

        backward[slot] = logsumexp(terms, axis=1)

    marginals = np.exp(
        forward + backward - log_partition
    )

    require(
        np.allclose(
            marginals.sum(axis=1),
            1.0,
            atol=1e-8,
            rtol=0.0,
        ),
        "Legalne marginalne raspodele nisu normalizovane.",
    )

    return {
        "path": path,
        "log_partition": log_partition,
        "log_mode_probability": float(
            values[last] - log_partition
        ),
        "marginals": marginals,
    }


def predict(unitaries, data, time, window):
    states = input_states(
        data,
        np.array([time]),
        window,
    )[0]

    initial_states = forecast_amplitudes(unitaries, states)
    solution = solve_vqe(initial_states)

    unary = legal_log_unary(solution["amplitudes"])
    distribution = legal_distribution(unary)
    path = distribution["path"]

    require(
        len(set(path)) == K
        and np.all(np.diff(path) > 0)
        and np.all((path >= 0) & (path < N)),
        "Nelegalan izlaz optimizacije.",
    )

    require(
        distribution["log_partition"] <= 1e-8
        and distribution["log_mode_probability"] <= 1e-8,
        "Neispravna normalizacija zavrsne raspodele.",
    )

    return {
        "states": states,
        "initial_states": initial_states,
        "solution": solution,
        "distribution": distribution,
        "unary": unary,
    }


def evaluate(unitaries, data, times, window, label):
    hits = []
    losses = []
    log_masses = []
    gradient_failures = 0
    zero_targets = 0

    for index, time in enumerate(times, start=1):
        prediction = predict(
            unitaries,
            data,
            int(time),
            window,
        )

        distribution = prediction["distribution"]
        solution = prediction["solution"]
        target = data[time]

        target_log_mass = float(
            prediction["unary"][np.arange(K), target].sum()
        )

        loss = (
            distribution["log_partition"]
            - target_log_mass
        )

        require(
            not np.isnan(loss) and loss >= -1e-8,
            "Neispravan evaluacioni NLL.",
        )

        if np.isposinf(loss):
            zero_targets += 1

        losses.append(loss)
        log_masses.append(distribution["log_partition"])

        hits.append(
            len(set(distribution["path"]) & set(target))
        )

        if solution["gradient_norm"] > VQE_GRADIENT_TOLERANCE:
            gradient_failures += 1

        if (
            index == 1
            or index % 25 == 0
            or index == len(times)
        ):
            print(
                label,
                index,
                "/",
                len(times),
                flush=True,
            )

    require(len(hits) > 0, "Prazan evaluacioni skup.")

    return {
        "nll": float(np.mean(losses)),
        "hits": np.asarray(hits, dtype=np.int64),
        "mean_log_mass": float(np.mean(log_masses)),
        "gradient_failures": gradient_failures,
        "zero_targets": zero_targets,
    }


def print_metrics(label, metrics):
    print(label)
    print("Kola:", len(metrics["hits"]))
    print("Srednji NLL:", f'{metrics["nll"]:.8f}')
    print(
        "Srednji broj pogodaka:",
        f'{metrics["hits"].mean():.6f}',
    )
    print(
        "Srednji log legalne mase:",
        f'{metrics["mean_log_mass"]:.8f}',
    )
    print("Nulte ciljne verovatnoce:", metrics["zero_targets"])
    print(
        "Kola bez dostignutog praga gradijenta:",
        metrics["gradient_failures"],
    )

    histogram = np.bincount(
        metrics["hits"],
        minlength=K + 1,
    )

    for count, occurrences in enumerate(histogram):
        print(
            "Pogodaka:",
            count,
            "kola:",
            int(occurrences),
        )


def verify_qiskit(unitaries, prediction):
    parameters = prediction["solution"]["parameters"].reshape(
        K,
        PARAMETERS_PER_REGISTER,
    )

    for slot in range(K):
        circuit = QuantumCircuit(QUBITS)

        circuit.initialize(
            prediction["states"][slot],
            range(QUBITS),
        )
        circuit.unitary(
            unitaries[slot],
            range(QUBITS),
        )

        for qubit in range(QUBITS):
            circuit.ry(
                float(parameters[slot, qubit]),
                qubit,
            )

        for control in range(QUBITS - 1):
            circuit.cx(control, control + 1)

        for qubit in range(QUBITS):
            circuit.ry(
                float(parameters[slot, QUBITS + qubit]),
                qubit,
            )

        qiskit_amplitudes = np.asarray(
            Statevector.from_instruction(circuit).data
        )

        numerical_amplitudes = (
            prediction["solution"]["amplitudes"][slot]
        )

        require(
            np.allclose(
                qiskit_amplitudes,
                numerical_amplitudes,
                atol=2e-10,
                rtol=0.0,
            ),
            "VQE amplitude se ne slazu sa Qiskitom.",
        )

    print(
        "Provera zavrsnih amplituda prema Qiskitu je prosla."
    )


def verify_mathematics(data):
    unary = np.sin(
        np.arange(15, dtype=float).reshape(3, 5) + 1.0
    )

    distribution = legal_distribution(unary)

    combinations = list(
        itertools.combinations(range(5), 3)
    )

    scores = np.array(
        [
            sum(
                unary[slot, combination[slot]]
                for slot in range(3)
            )
            for combination in combinations
        ]
    )

    require(
        np.isclose(
            distribution["log_partition"],
            logsumexp(scores),
            atol=1e-12,
            rtol=0.0,
        ),
        "Iscrpna provera normalizacije nije prosla.",
    )

    require(
        tuple(distribution["path"])
        == combinations[int(np.argmax(scores))],
        "Iscrpna provera maksimuma nije prosla.",
    )

    initial_states = input_states(
        data,
        np.array([1]),
        WINDOWS[0],
    )[0]

    item_values = probabilities(initial_states)
    local_energy = build_hamiltonian(item_values)

    parameters = 0.15 * np.sin(
        np.arange(K * PARAMETERS_PER_REGISTER) + 1.0
    )

    _, gradient = vqe_objective(
        parameters,
        initial_states,
        local_energy,
    )

    step = 1e-6

    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[index] += step
        minus[index] -= step

        plus_energy = vqe_objective(
            plus,
            initial_states,
            local_energy,
        )[0]

        minus_energy = vqe_objective(
            minus,
            initial_states,
            local_energy,
        )[0]

        numerical = (
            plus_energy - minus_energy
        ) / (2.0 * step)

        require(
            np.isclose(
                numerical,
                gradient[index],
                atol=2e-6,
                rtol=2e-4,
            ),
            f"VQE gradijent nije potvrdjen za parametar {index}.",
        )

    print(
        "Provere normalizacije, maksimuma i VQE gradijenta su prosle."
    )


def process_dataset(dataset):
    data = dataset["data"]
    total = len(data) - 1

    train_count = int(TRAIN_FRACTION * total)
    validation_end = int(
        VALIDATION_END_FRACTION * total
    )

    validation_times = np.arange(
        train_count + 1,
        validation_end + 1,
    )
    test_times = np.arange(
        validation_end + 1,
        len(data),
    )

    require(
        0 < train_count < validation_end < total,
        "Neispravna hronoloska podela.",
    )

    print()
    print("CSV:", dataset["path"].name)
    print("Redova:", len(data))
    print("SHA256:", dataset["digest"])
    print("Trening prelaza:", train_count)
    print("Validacionih kola:", len(validation_times))
    print("Test kola:", len(test_times))

    best_settings = None
    best_nll = np.inf

    for window in WINDOWS:
        for regularization in REGULARIZATIONS:
            print()
            print(
                "Kandidat:",
                "prozor",
                window,
                "regularizacija",
                regularization,
            )

            unitaries = fit_unitaries(
                data,
                train_count,
                window,
                regularization,
            )

            try:
                validation = evaluate(
                    unitaries,
                    data,
                    validation_times,
                    window,
                    "VALIDACIJA",
                )
            except DistributionError as error:
                print(
                    "Kandidat nema definisan legalan izlaz."
                )
                print(str(error))
                continue

            print_metrics(
                "VALIDACIONI REZULTAT",
                validation,
            )

            if (
                np.isfinite(validation["nll"])
                and validation["nll"] < best_nll
            ):
                best_nll = validation["nll"]
                best_settings = (
                    window,
                    regularization,
                )

    require(
        best_settings is not None,
        "Nijedan kandidat nema konacan validacioni NLL.",
    )

    window, regularization = best_settings

    print()
    print("Izabrani prozor:", window)
    print(
        "Izabrana regularizacija:",
        regularization,
    )

    test_unitaries = fit_unitaries(
        data,
        validation_end,
        window,
        regularization,
    )

    test_metrics = evaluate(
        test_unitaries,
        data,
        test_times,
        window,
        "TEST",
    )
    print_metrics(
        "ODVOJENI TEST CELOG POSTUPKA",
        test_metrics,
    )

    # Test ne bira hiperparametre.
    # Ranija test kola su istorija kasnijih test kola.
    # Operator ostaje zamrznut tokom testa.

    final_unitaries = fit_unitaries(
        data,
        total,
        window,
        regularization,
    )

    print(
        "Zavrsno ucenje koristi istorijskih prelaza:",
        total,
    )

    prediction = predict(
        final_unitaries,
        data,
        len(data),
        window,
    )

    verify_qiskit(final_unitaries, prediction)

    solution = prediction["solution"]
    distribution = prediction["distribution"]

    combination = tuple(
        int(value + 1)
        for value in distribution["path"]
    )

    print(
        "Pocetna VQE energija:",
        f'{solution["initial_energy"]:.10f}',
    )
    print(
        "Zavrsna VQE energija:",
        f'{solution["energy"]:.10f}',
    )
    print("VQE status:", solution["message"])
    print(
        "VQE norma gradijenta:",
        f'{solution["gradient_norm"]:.8g}',
    )
    print(
        "Log legalne mase:",
        f'{distribution["log_partition"]:.8f}',
    )
    print(
        "Log modelske verovatnoce NEXT:",
        f'{distribution["log_mode_probability"]:.8f}',
    )

    if solution["gradient_norm"] > VQE_GRADIENT_TOLERANCE:
        print(
            "UPOZORENJE: VQE kriterijum gradijenta nije dostignut."
        )

    if distribution["log_partition"] < math.log(0.5):
        print(
            "UPOZORENJE: vise od pola sirove mase je nelegalno."
        )

    current_run = 1
    longest_run = 1

    for difference in np.diff(combination):
        if difference == 1:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )
        else:
            current_run = 1

    if longest_run >= 5:
        print(
            "UPOZORENJE: najmanje pet uzastopnih brojeva zahteva pregled."
        )

    print(
        "NEXT:",
        " ".join(map(str, combination)),
    )

    return {
        "name": dataset["path"].name,
        "digest": dataset["digest"],
        "next": combination,
        "probabilities": probabilities(
            solution["amplitudes"]
        ),
        "marginals": distribution["marginals"],
    }


def main():
    print("Quantum MDKP + MIS")
    print("Qiskit:", qiskit.__version__)
    print(
        "Zajednicki Hamiltonijan i VQE minimizacija energije."
    )
    print(
        "Sedam faktorizovanih registara sa po sest kubita."
    )
    print(
        "Prediktivna dopuna: unitarna regresija amplituda."
    )
    print(
        "Validacija i test izvode ceo VQE postupak za svako kolo."
    )
    print(
        "Ova provera moze zahtevati znatno vreme."
    )

    datasets = [
        load_csv(path)
        for path in FILES
    ]

    verify_mathematics(datasets[0]["data"])

    results = [
        process_dataset(dataset)
        for dataset in datasets
    ]

    for first, second in itertools.combinations(results, 2):
        if first["digest"] == second["digest"]:
            print(
                "UPOZORENJE: dva ulaza imaju isti SHA256."
            )
            print(first["name"])
            print(second["name"])

        if first["next"] == second["next"]:
            print()
            print(
                "PROVERA ISTIH NEXT KOMBINACIJA"
            )
            print(first["name"])
            print(second["name"])
            print(
                "Najveca razlika Bornovih verovatnoca:",
                float(np.max(np.abs(
                    first["probabilities"]
                    - second["probabilities"]
                ))),
            )
            print(
                "Najveca razlika legalnih marginalnih raspodela:",
                float(np.max(np.abs(
                    first["marginals"]
                    - second["marginals"]
                ))),
            )
            print(
                "Isti NEXT zahteva pregled dijagnostika."
            )

    print()
    print("ZAVRSNE NEXT KOMBINACIJE")

    for result in results:
        print(result["name"])
        print(
            "NEXT:",
            " ".join(map(str, result["next"])),
        )


if __name__ == "__main__":
    main()



"""
Quantum MDKP + MIS
Qiskit: 1.4.4
Zajednicki Hamiltonijan i VQE minimizacija energije.
Sedam faktorizovanih registara sa po sest kubita.
Prediktivna dopuna: unitarna regresija amplituda.
Validacija i test izvode ceo VQE postupak za svako kolo.
Ova provera moze zahtevati znatno vreme.
Provere normalizacije, maksimuma i VQE gradijenta su prosle.

CSV: loto7_4716_k89.csv
Redova: 4716
SHA256: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Trening prelaza: 3300
Validacionih kola: 707
Test kola: 708

Kandidat: prozor 1 regularizacija 0.01
VALIDACIJA 1 / 707
VALIDACIJA 25 / 707
VALIDACIJA 50 / 707
VALIDACIJA 75 / 707
VALIDACIJA 100 / 707
VALIDACIJA 125 / 707
VALIDACIJA 150 / 707
VALIDACIJA 175 / 707
VALIDACIJA 200 / 707
VALIDACIJA 225 / 707
VALIDACIJA 250 / 707
VALIDACIJA 275 / 707
VALIDACIJA 300 / 707
VALIDACIJA 325 / 707
VALIDACIJA 350 / 707
VALIDACIJA 375 / 707
VALIDACIJA 400 / 707
VALIDACIJA 425 / 707
VALIDACIJA 450 / 707
VALIDACIJA 475 / 707
VALIDACIJA 500 / 707
VALIDACIJA 525 / 707
VALIDACIJA 550 / 707
VALIDACIJA 575 / 707
VALIDACIJA 600 / 707
VALIDACIJA 625 / 707
VALIDACIJA 650 / 707
VALIDACIJA 675 / 707
VALIDACIJA 700 / 707
VALIDACIJA 707 / 707
VALIDACIONI REZULTAT
Kola: 707
Srednji NLL: 51.45113730
Srednji broj pogodaka: 1.234795
Srednji log legalne mase: -0.50074817
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 707
Pogodaka: 0 kola: 166
Pogodaka: 1 kola: 299
Pogodaka: 2 kola: 164
Pogodaka: 3 kola: 67
Pogodaka: 4 kola: 10
Pogodaka: 5 kola: 1
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Kandidat: prozor 1 regularizacija 0.1
VALIDACIJA 1 / 707
VALIDACIJA 25 / 707
VALIDACIJA 50 / 707
VALIDACIJA 75 / 707
VALIDACIJA 100 / 707
VALIDACIJA 125 / 707
VALIDACIJA 150 / 707
VALIDACIJA 175 / 707
VALIDACIJA 200 / 707
VALIDACIJA 225 / 707
VALIDACIJA 250 / 707
VALIDACIJA 275 / 707
VALIDACIJA 300 / 707
VALIDACIJA 325 / 707
VALIDACIJA 350 / 707
VALIDACIJA 375 / 707
VALIDACIJA 400 / 707
VALIDACIJA 425 / 707
VALIDACIJA 450 / 707
VALIDACIJA 475 / 707
VALIDACIJA 500 / 707
VALIDACIJA 525 / 707
VALIDACIJA 550 / 707
VALIDACIJA 575 / 707
VALIDACIJA 600 / 707
VALIDACIJA 625 / 707
VALIDACIJA 650 / 707
VALIDACIJA 675 / 707
VALIDACIJA 700 / 707
VALIDACIJA 707 / 707
VALIDACIONI REZULTAT
Kola: 707
Srednji NLL: 40.36242529
Srednji broj pogodaka: 1.254597
Srednji log legalne mase: -1.29424390
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 707
Pogodaka: 0 kola: 149
Pogodaka: 1 kola: 297
Pogodaka: 2 kola: 204
Pogodaka: 3 kola: 47
Pogodaka: 4 kola: 9
Pogodaka: 5 kola: 1
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Kandidat: prozor 3 regularizacija 0.01
VALIDACIJA 1 / 707
VALIDACIJA 25 / 707
VALIDACIJA 50 / 707
VALIDACIJA 75 / 707
VALIDACIJA 100 / 707
VALIDACIJA 125 / 707
VALIDACIJA 150 / 707
VALIDACIJA 175 / 707
VALIDACIJA 200 / 707
VALIDACIJA 225 / 707
VALIDACIJA 250 / 707
VALIDACIJA 275 / 707
VALIDACIJA 300 / 707
VALIDACIJA 325 / 707
VALIDACIJA 350 / 707
VALIDACIJA 375 / 707
VALIDACIJA 400 / 707
VALIDACIJA 425 / 707
VALIDACIJA 450 / 707
VALIDACIJA 475 / 707
VALIDACIJA 500 / 707
VALIDACIJA 525 / 707
VALIDACIJA 550 / 707
VALIDACIJA 575 / 707
VALIDACIJA 600 / 707
VALIDACIJA 625 / 707
VALIDACIJA 650 / 707
VALIDACIJA 675 / 707
VALIDACIJA 700 / 707
VALIDACIJA 707 / 707
VALIDACIONI REZULTAT
Kola: 707
Srednji NLL: 45.68275924
Srednji broj pogodaka: 1.278642
Srednji log legalne mase: -0.63327638
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 707
Pogodaka: 0 kola: 145
Pogodaka: 1 kola: 304
Pogodaka: 2 kola: 185
Pogodaka: 3 kola: 64
Pogodaka: 4 kola: 7
Pogodaka: 5 kola: 2
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Kandidat: prozor 3 regularizacija 0.1
VALIDACIJA 1 / 707
VALIDACIJA 25 / 707
VALIDACIJA 50 / 707
VALIDACIJA 75 / 707
VALIDACIJA 100 / 707
VALIDACIJA 125 / 707
VALIDACIJA 150 / 707
VALIDACIJA 175 / 707
VALIDACIJA 200 / 707
VALIDACIJA 225 / 707
VALIDACIJA 250 / 707
VALIDACIJA 275 / 707
VALIDACIJA 300 / 707
VALIDACIJA 325 / 707
VALIDACIJA 350 / 707
VALIDACIJA 375 / 707
VALIDACIJA 400 / 707
VALIDACIJA 425 / 707
VALIDACIJA 450 / 707
VALIDACIJA 475 / 707
VALIDACIJA 500 / 707
VALIDACIJA 525 / 707
VALIDACIJA 550 / 707
VALIDACIJA 575 / 707
VALIDACIJA 600 / 707
VALIDACIJA 625 / 707
VALIDACIJA 650 / 707
VALIDACIJA 675 / 707
VALIDACIJA 700 / 707
VALIDACIJA 707 / 707
VALIDACIONI REZULTAT
Kola: 707
Srednji NLL: 35.69524871
Srednji broj pogodaka: 1.246110
Srednji log legalne mase: -1.69325826
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 707
Pogodaka: 0 kola: 155
Pogodaka: 1 kola: 299
Pogodaka: 2 kola: 187
Pogodaka: 3 kola: 56
Pogodaka: 4 kola: 10
Pogodaka: 5 kola: 0
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Izabrani prozor: 3
Izabrana regularizacija: 0.1
TEST 1 / 708
TEST 25 / 708
TEST 50 / 708
TEST 75 / 708
TEST 100 / 708
TEST 125 / 708
TEST 150 / 708
TEST 175 / 708
TEST 200 / 708
TEST 225 / 708
TEST 250 / 708
TEST 275 / 708
TEST 300 / 708
TEST 325 / 708
TEST 350 / 708
TEST 375 / 708
TEST 400 / 708
TEST 425 / 708
TEST 450 / 708
TEST 475 / 708
TEST 500 / 708
TEST 525 / 708
TEST 550 / 708
TEST 575 / 708
TEST 600 / 708
TEST 625 / 708
TEST 650 / 708
TEST 675 / 708
TEST 700 / 708
TEST 708 / 708
ODVOJENI TEST CELOG POSTUPKA
Kola: 708
Srednji NLL: 35.46883679
Srednji broj pogodaka: 1.264124
Srednji log legalne mase: -1.69863230
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 708
Pogodaka: 0 kola: 158
Pogodaka: 1 kola: 282
Pogodaka: 2 kola: 203
Pogodaka: 3 kola: 54
Pogodaka: 4 kola: 10
Pogodaka: 5 kola: 1
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0
Zavrsno ucenje koristi istorijskih prelaza: 4715
Provera zavrsnih amplituda prema Qiskitu je prosla.
Pocetna VQE energija: 7.9457725356
Zavrsna VQE energija: 2.2274223101
VQE status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
VQE norma gradijenta: 0.0014309178
Log legalne mase: -1.32800564
Log modelske verovatnoce NEXT: -3.76096823
UPOZORENJE: VQE kriterijum gradijenta nije dostignut.
UPOZORENJE: vise od pola sirove mase je nelegalno.
NEXT: 1 2 5 8 13 29 31

CSV: loto7_4716_k89_loto_2980.csv
Redova: 2979
SHA256: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Trening prelaza: 2084
Validacionih kola: 447
Test kola: 447

Kandidat: prozor 1 regularizacija 0.01
VALIDACIJA 1 / 447
VALIDACIJA 25 / 447
VALIDACIJA 50 / 447
VALIDACIJA 75 / 447
VALIDACIJA 100 / 447
VALIDACIJA 125 / 447
VALIDACIJA 150 / 447
VALIDACIJA 175 / 447
VALIDACIJA 200 / 447
VALIDACIJA 225 / 447
VALIDACIJA 250 / 447
VALIDACIJA 275 / 447
VALIDACIJA 300 / 447
VALIDACIJA 325 / 447
VALIDACIJA 350 / 447
VALIDACIJA 375 / 447
VALIDACIJA 400 / 447
VALIDACIJA 425 / 447
VALIDACIJA 447 / 447
VALIDACIONI REZULTAT
Kola: 447
Srednji NLL: 51.25598080
Srednji broj pogodaka: 1.217002
Srednji log legalne mase: -0.49519846
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 447
Pogodaka: 0 kola: 102
Pogodaka: 1 kola: 184
Pogodaka: 2 kola: 128
Pogodaka: 3 kola: 28
Pogodaka: 4 kola: 5
Pogodaka: 5 kola: 0
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Kandidat: prozor 1 regularizacija 0.1
VALIDACIJA 1 / 447
VALIDACIJA 25 / 447
VALIDACIJA 50 / 447
VALIDACIJA 75 / 447
VALIDACIJA 100 / 447
VALIDACIJA 125 / 447
VALIDACIJA 150 / 447
VALIDACIJA 175 / 447
VALIDACIJA 200 / 447
VALIDACIJA 225 / 447
VALIDACIJA 250 / 447
VALIDACIJA 275 / 447
VALIDACIJA 300 / 447
VALIDACIJA 325 / 447
VALIDACIJA 350 / 447
VALIDACIJA 375 / 447
VALIDACIJA 400 / 447
VALIDACIJA 425 / 447
VALIDACIJA 447 / 447
VALIDACIONI REZULTAT
Kola: 447
Srednji NLL: 39.99877395
Srednji broj pogodaka: 1.261745
Srednji log legalne mase: -1.30569169
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 447
Pogodaka: 0 kola: 104
Pogodaka: 1 kola: 168
Pogodaka: 2 kola: 135
Pogodaka: 3 kola: 35
Pogodaka: 4 kola: 4
Pogodaka: 5 kola: 1
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Kandidat: prozor 3 regularizacija 0.01
VALIDACIJA 1 / 447
VALIDACIJA 25 / 447
VALIDACIJA 50 / 447
VALIDACIJA 75 / 447
VALIDACIJA 100 / 447
VALIDACIJA 125 / 447
VALIDACIJA 150 / 447
VALIDACIJA 175 / 447
VALIDACIJA 200 / 447
VALIDACIJA 225 / 447
VALIDACIJA 250 / 447
VALIDACIJA 275 / 447
VALIDACIJA 300 / 447
VALIDACIJA 325 / 447
VALIDACIJA 350 / 447
VALIDACIJA 375 / 447
VALIDACIJA 400 / 447
VALIDACIJA 425 / 447
VALIDACIJA 447 / 447
VALIDACIONI REZULTAT
Kola: 447
Srednji NLL: 45.18281455
Srednji broj pogodaka: 1.250559
Srednji log legalne mase: -0.62679365
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 447
Pogodaka: 0 kola: 98
Pogodaka: 1 kola: 186
Pogodaka: 2 kola: 120
Pogodaka: 3 kola: 39
Pogodaka: 4 kola: 4
Pogodaka: 5 kola: 0
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Kandidat: prozor 3 regularizacija 0.1
VALIDACIJA 1 / 447
VALIDACIJA 25 / 447
VALIDACIJA 50 / 447
VALIDACIJA 75 / 447
VALIDACIJA 100 / 447
VALIDACIJA 125 / 447
VALIDACIJA 150 / 447
VALIDACIJA 175 / 447
VALIDACIJA 200 / 447
VALIDACIJA 225 / 447
VALIDACIJA 250 / 447
VALIDACIJA 275 / 447
VALIDACIJA 300 / 447
VALIDACIJA 325 / 447
VALIDACIJA 350 / 447
VALIDACIJA 375 / 447
VALIDACIJA 400 / 447
VALIDACIJA 425 / 447
VALIDACIJA 447 / 447
VALIDACIONI REZULTAT
Kola: 447
Srednji NLL: 35.77106136
Srednji broj pogodaka: 1.319911
Srednji log legalne mase: -1.66283391
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 447
Pogodaka: 0 kola: 99
Pogodaka: 1 kola: 173
Pogodaka: 2 kola: 120
Pogodaka: 3 kola: 44
Pogodaka: 4 kola: 10
Pogodaka: 5 kola: 1
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Izabrani prozor: 3
Izabrana regularizacija: 0.1
TEST 1 / 447
TEST 25 / 447
TEST 50 / 447
TEST 75 / 447
TEST 100 / 447
TEST 125 / 447
TEST 150 / 447
TEST 175 / 447
TEST 200 / 447
TEST 225 / 447
TEST 250 / 447
TEST 275 / 447
TEST 300 / 447
TEST 325 / 447
TEST 350 / 447
TEST 375 / 447
TEST 400 / 447
TEST 425 / 447
TEST 447 / 447
ODVOJENI TEST CELOG POSTUPKA
Kola: 447
Srednji NLL: 35.16786119
Srednji broj pogodaka: 1.272931
Srednji log legalne mase: -1.70465195
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 447
Pogodaka: 0 kola: 97
Pogodaka: 1 kola: 173
Pogodaka: 2 kola: 139
Pogodaka: 3 kola: 34
Pogodaka: 4 kola: 4
Pogodaka: 5 kola: 0
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0
Zavrsno ucenje koristi istorijskih prelaza: 2978
Provera zavrsnih amplituda prema Qiskitu je prosla.
Pocetna VQE energija: 8.1792684391
Zavrsna VQE energija: 2.2128893896
VQE status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
VQE norma gradijenta: 0.0085364544
Log legalne mase: -1.30953937
Log modelske verovatnoce NEXT: -3.03938371
UPOZORENJE: VQE kriterijum gradijenta nije dostignut.
UPOZORENJE: vise od pola sirove mase je nelegalno.
NEXT: 1 4 5 10 13 29 35

CSV: loto7_4716_k89_loto_plus_1736.csv
Redova: 1735
SHA256: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Trening prelaza: 1213
Validacionih kola: 260
Test kola: 261

Kandidat: prozor 1 regularizacija 0.01
VALIDACIJA 1 / 260
VALIDACIJA 25 / 260
VALIDACIJA 50 / 260
VALIDACIJA 75 / 260
VALIDACIJA 100 / 260
VALIDACIJA 125 / 260
VALIDACIJA 150 / 260
VALIDACIJA 175 / 260
VALIDACIJA 200 / 260
VALIDACIJA 225 / 260
VALIDACIJA 250 / 260
VALIDACIJA 260 / 260
VALIDACIONI REZULTAT
Kola: 260
Srednji NLL: 50.64516986
Srednji broj pogodaka: 1.173077
Srednji log legalne mase: -0.50556121
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 260
Pogodaka: 0 kola: 63
Pogodaka: 1 kola: 113
Pogodaka: 2 kola: 62
Pogodaka: 3 kola: 20
Pogodaka: 4 kola: 2
Pogodaka: 5 kola: 0
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Kandidat: prozor 1 regularizacija 0.1
VALIDACIJA 1 / 260
VALIDACIJA 25 / 260
VALIDACIJA 50 / 260
VALIDACIJA 75 / 260
VALIDACIJA 100 / 260
VALIDACIJA 125 / 260
VALIDACIJA 150 / 260
VALIDACIJA 175 / 260
VALIDACIJA 200 / 260
VALIDACIJA 225 / 260
VALIDACIJA 250 / 260
VALIDACIJA 260 / 260
VALIDACIONI REZULTAT
Kola: 260
Srednji NLL: 40.38431510
Srednji broj pogodaka: 1.169231
Srednji log legalne mase: -1.30836535
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 260
Pogodaka: 0 kola: 66
Pogodaka: 1 kola: 102
Pogodaka: 2 kola: 77
Pogodaka: 3 kola: 12
Pogodaka: 4 kola: 3
Pogodaka: 5 kola: 0
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Kandidat: prozor 3 regularizacija 0.01
VALIDACIJA 1 / 260
VALIDACIJA 25 / 260
VALIDACIJA 50 / 260
VALIDACIJA 75 / 260
VALIDACIJA 100 / 260
VALIDACIJA 125 / 260
VALIDACIJA 150 / 260
VALIDACIJA 175 / 260
VALIDACIJA 200 / 260
VALIDACIJA 225 / 260
VALIDACIJA 250 / 260
VALIDACIJA 260 / 260
VALIDACIONI REZULTAT
Kola: 260
Srednji NLL: 45.31223976
Srednji broj pogodaka: 1.215385
Srednji log legalne mase: -0.61185482
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 260
Pogodaka: 0 kola: 58
Pogodaka: 1 kola: 115
Pogodaka: 2 kola: 62
Pogodaka: 3 kola: 23
Pogodaka: 4 kola: 2
Pogodaka: 5 kola: 0
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Kandidat: prozor 3 regularizacija 0.1
VALIDACIJA 1 / 260
VALIDACIJA 25 / 260
VALIDACIJA 50 / 260
VALIDACIJA 75 / 260
VALIDACIJA 100 / 260
VALIDACIJA 125 / 260
VALIDACIJA 150 / 260
VALIDACIJA 175 / 260
VALIDACIJA 200 / 260
VALIDACIJA 225 / 260
VALIDACIJA 250 / 260
VALIDACIJA 260 / 260
VALIDACIONI REZULTAT
Kola: 260
Srednji NLL: 35.53867181
Srednji broj pogodaka: 1.296154
Srednji log legalne mase: -1.62728446
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 260
Pogodaka: 0 kola: 48
Pogodaka: 1 kola: 111
Pogodaka: 2 kola: 77
Pogodaka: 3 kola: 24
Pogodaka: 4 kola: 0
Pogodaka: 5 kola: 0
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0

Izabrani prozor: 3
Izabrana regularizacija: 0.1
TEST 1 / 261
TEST 25 / 261
TEST 50 / 261
TEST 75 / 261
TEST 100 / 261
TEST 125 / 261
TEST 150 / 261
TEST 175 / 261
TEST 200 / 261
TEST 225 / 261
TEST 250 / 261
TEST 261 / 261
ODVOJENI TEST CELOG POSTUPKA
Kola: 261
Srednji NLL: 35.30252305
Srednji broj pogodaka: 1.199234
Srednji log legalne mase: -1.67672821
Nulte ciljne verovatnoce: 0
Kola bez dostignutog praga gradijenta: 261
Pogodaka: 0 kola: 57
Pogodaka: 1 kola: 121
Pogodaka: 2 kola: 62
Pogodaka: 3 kola: 17
Pogodaka: 4 kola: 3
Pogodaka: 5 kola: 1
Pogodaka: 6 kola: 0
Pogodaka: 7 kola: 0
Zavrsno ucenje koristi istorijskih prelaza: 1734
Provera zavrsnih amplituda prema Qiskitu je prosla.
Pocetna VQE energija: 8.2341985956
Zavrsna VQE energija: 2.5971234892
VQE status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
VQE norma gradijenta: 0.01067128
Log legalne mase: -1.57933421
Log modelske verovatnoce NEXT: -3.67989376
UPOZORENJE: VQE kriterijum gradijenta nije dostignut.
UPOZORENJE: vise od pola sirove mase je nelegalno.
NEXT: 1 5 7 10 13 23 37



ZAVRSNE NEXT KOMBINACIJE

loto7_4716_k89.csv
NEXT: 1 2 5 8 13 29 31

loto7_4716_k89_loto_2980.csv
NEXT: 1 4 5 10 13 29 35

loto7_4716_k89_loto_plus_1736.csv
NEXT: 1 5 7 10 13 23 37
"""



"""
1. Multi-Dimensional Knapsack
Izbor predmeta sa najvećom ukupnom vrednošću uz više ograničenja. 
Svaki broj predstavlja jedan predmet. Zajednička optimizacija cele kombinacije. 

2. Maximum Independent Set
Izbor najvećeg skupa čvorova između kojih nema konfliktnih veza. Za Loto bi 39 brojeva bili čvorovi, a veze bi označavale zabranu zajedničkog izbora. 
Ažuriranje grafa posle svakog izbora. 

- VQE postupak i optimizaciju amplituda kvantnog stanja.
- Provera ograničenja i kvaliteta ciljne funkcije.
- Izbor NEXT prema najvećoj verovatnoći legalne kombinacije, uz proveru maksimuma.
- Knapsack deo bira tačno sedam brojeva.
- MIS deo sprečava duplikate pri izboru iz sedam registara. Veze predstavljaju stvarne konflikte izbora.
- Kvantni deo koristi Qiskit i amplitude na lokalnom simulatoru.

Knapsack deo bira po jedan broj iz sedam registara. MIS deo sprečava konflikte koji narušavaju rastuću kombinaciju. VQE koristi amplitude na lokalnom Qiskit simulatoru.
 
Kvantni deo je VQE optimizacija. 
Sedam registara se simulira faktorisano, sa zapletenošću unutar svakog registra.

fit_model() uči parametre kvantnih kola preko amplituda.
Validacija, ceo odvojeni test i NEXT koriste istu normalizovanu kvantnu raspodelu.
MDKP ograničenja obezbeđuju jedan izbor po registru. MIS Hamiltonijan kažnjava konflikte.
Obuka koristi regresiju kvantne raspodele uz Hamiltonijan ograničenja. 
Sedam registara ima proizvoljno početno stanje. 
Uslovljavanje legalnošću i globalni maksimum računaju se klasično.


- Logaritmi se računaju direktno iz amplituda.
- Gradijent se računa bez deljenja kvadratom vrlo male amplitude.
- Nedozvoljen korak optimizatora se odbacuje. Ne vraća se lažni nulti gradijent.
- Nulta verovatnoća stvarnog kola daje beskonačan NLL.
- Sva tri ulaza proveravaju se pre obuke.


QUBO ciljna funkcija, Hamiltonijan, parametrizovano kvantno kolo i VQE minimizaciju energije.

Svaki čvor predstavlja izbor broja u jednom registru. 
MDKP deo zahteva po jedan izbor iz sedam grupa. 
MIS deo zabranjuje konfliktne izbore. 

Kompaktno kodiranje zadovoljava grupna ograničenja. 
VQE optimizuje preostalu energiju. 
Dodatna kvantna regresija amplituda uči težine \(v_i\) iz istorijskih prelaza. 
VQE ostaje približan optimizator sa sedam faktorizovanih šestokubitnih registara. 
Program prikazuje konvergenciju i legalnu masu.
"""



"""
https://medium.com/@_monitsharma/solving-the-multi-dimensional-knapsack-problem-a-classically-hard-benchmark-for-quantum-2610e955d114

https://medium.com/@_monitsharma/solving-the-maximum-independent-set-problem-a-classically-hard-benchmark-for-quantum-optimization-5d8e9c267052
"""
