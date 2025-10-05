import sys
sys.path.append('../shelter')

from shelter import *

# Example usage
if __name__ == "__main__":
    sys = System("Kepler-16")
    star_a = Star("Kepler-16A")
    star_b = Star("Kepler-16B")
    planet = Planet("Kepler-16b")

    sys.add_star(star_a)
    sys.add_star(star_b)
    sys.add_planet(planet)

    star_a.add_planet(planet)
    star_b.add_planet(planet)

    planet.set_param("mass", 5.3, upper=0.2, lower=0.3, aliases=["m"])
    print(planet.mass.value)   # 5.3
    print(planet.mass.upper)   # 0.2

    # Assign posterior samples
    planet.set_posterior_samples("mass", np.random.normal(1.5, 0.1, size=1000))

    print(planet.mass.distribution)       # Distribution(...)
    print(np.std(planet.mass.samples))    # std of samples

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total running time: {elapsed_time:.8f} seconds")

    #print(query_archive("HD 133131", save=True, overwrite=True))
    system = setup_system("Kepler-32", save=False)
    print(system.name)

