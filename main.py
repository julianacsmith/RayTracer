import numpy as np
import sys
import matplotlib.pyplot as plt

# Holds New Spheres specified in files
class Sphere:
    def __init__(self):
        self.center = []
        self.radius = 0.0
        self.kd = 0.0
        self.ks = 0.0
        self.ka = 0.0
        self.od = []
        self.os = []
        self.kgls = 0.0
        self.refl = 0.0

    # Sets the variables for each specific sphere to be put into the objects[]
    # in the RayTracer itself
    def set_sphere(self, lines):
        new_line = False
        while lines and not new_line:
            line = lines[0]
            info = line.split(" ")
            if len(info) > 1:
                if info[2] == "Center":
                    self.center = np.array([float(info[3]), float(info[4]), float(info[5])])
                elif info[2] == "Radius":
                    self.radius = float(info[3])
                elif info[2] == "Kd":
                    self.kd = float(info[3])
                elif info[2] == "Ks":
                    self.ks = float(info[3])
                elif info[2] == "Ka":
                    self.ka = float(info[3])
                elif info[2] == "Od":
                    self.od = np.array([float(info[3]), float(info[4]), float(info[5])])
                elif info[2] == "Os":
                    self.os = np.array([float(info[3]), float(info[4]), float(info[5])])
                elif info[2] == "Kgls":
                    self.kgls = float(info[3])
                elif info[2] == "Refl":
                    self.refl = float(info[3])
            else:
                new_line = True
            lines.remove(line)


class Triangle:
    def __init__(self):
        self.v1 = []
        self.v2 = []
        self.v3 = []
        self.kd = 0.0
        self.ks = 0.0
        self.ka = 0.0
        self.od = []
        self.os = []
        self.kgls = 0.0
        self.refl = 0.0

    def set_triangle(self, lines):
        new_line = False
        vert_num = 1
        while lines and not new_line:
            line = lines[0]
            info = line.split(" ")
            if len(info) > 1:
                if info[2] == "Kd":
                    self.kd = float(info[3])
                elif info[2] == "Ks":
                    self.ks = float(info[3])
                elif info[2] == "Ka":
                    self.ka = float(info[3])
                elif info[2] == "Od":
                    self.od = np.array([float(info[3]), float(info[4]), float(info[5])])
                elif info[2] == "Os":
                    self.os = np.array([float(info[3]), float(info[4]), float(info[5])])
                elif info[2] == "Kgls":
                    self.kgls = float(info[3])
                elif info[2] == "Refl":
                    self.refl = float(info[3])
                elif vert_num == 1:
                    self.v1 = np.array([float(info[2]), float(info[3]), float(info[4])])
                    vert_num += 1
                elif vert_num == 2:
                    self.v2 = np.array([float(info[2]), float(info[3]), float(info[4])])
                    vert_num += 1
                elif vert_num == 3:
                    self.v3 = np.array([float(info[2]), float(info[3]), float(info[4])])
                    vert_num += 1
            else:
                new_line = True
            lines.remove(line)

# Holds the origin and direction of a Ray
class Ray:
    def __init__(self):
        self.origin = []
        self.direction = []


# Actually does the ray tracing
class RayTracer:
    def __init__(self):
        self.cameraLookAt = []
        self.cameraLookFrom = []
        self.cameraLookUp = []
        self.fieldOfView = 0
        self.directionToLight = []
        self.lightColor = []
        self.ambientLight = []
        self.backgroundColor = []
        self.spheres = []
        self.triangles = []
        self.image_width = 500
        self.image_height = 500
        # Sets default image to all black to be written over later
        self.image = np.zeros((self.image_height, self.image_width, 3))
        self.max_bounce = 1
        self.curr_bounce = 0

    # Reading in a specified image file
    def read_image(self, file_name):
        file = open(file_name, "r")
        lines = file.readlines()

        # Set all the appropriate variable and objects
        while lines:
            line = lines[0]
            info = line.split(" ")
            if info[0] != "\n" and info[0].find("#") == -1:
                if info[0] == "CameraLookAt":
                    self.cameraLookAt = np.array([int(info[1]), int(info[2]), int(info[3])])
                    lines.remove(line)
                elif info[0] == "CameraLookFrom":
                    self.cameraLookFrom = np.array([int(info[1]), int(info[2]), int(info[3])])
                    lines.remove(line)
                elif info[0] == "CameraLookUp":
                    self.cameraLookUp = np.array([int(info[1]), int(info[2]), int(info[3])])
                    lines.remove(line)
                elif info[0] == "FieldOfView":
                    self.fieldOfView = int(info[1])
                    lines.remove(line)
                elif info[0] == "DirectionToLight":
                    self.directionToLight = np.array([float(info[1]), float(info[2]), float(info[3])])
                    lines.remove(line)
                elif info[0] == "LightColor":
                    self.lightColor = np.array([float(info[1]), float(info[2]), float(info[3])])
                    lines.remove(line)
                elif info[0] == "AmbientLight":
                    self.ambientLight = np.array([float(info[1]), float(info[2]), float(info[3])])
                    lines.remove(line)
                elif info[0] == "BackgroundColor":
                    self.backgroundColor = np.array([float(info[1]), float(info[2]), float(info[3])])
                    lines.remove(line)
                elif "Sphere" in info[0]:
                    lines.remove(line)
                    sphere = Sphere()
                    sphere.set_sphere(lines)
                    self.spheres.append(sphere)
                elif "Triangle" in info[0]:
                    lines.remove(line)
                    triangle = Triangle()
                    triangle.set_triangle(lines)
                    self.triangles.append(triangle)
            else:   # If a comment or empty line
                lines.remove(line)

    # Normalize a given vector
    def normalize(self, vec):
        return vec / np.linalg.norm(vec)

    # Determines if a ray intersects and specified sphere
    def sphere_intersection(self, sphere, ray):
        # Calculate b and c of the quadratic formula
        # a will equal 1 so its negligible
        b = 2 * (ray.direction[0] * ray.origin[0] - ray.direction[0] * sphere.center[0]
                 + ray.direction[1] * ray.origin[1] - ray.direction[1] * sphere.center[1]
                 + ray.direction[2] * ray.origin[2] - ray.direction[2] * sphere.center[2])
        c = ((ray.origin[0] ** 2 - 2 * ray.origin[0] * sphere.center[0] + sphere.center[0] ** 2)
             + (ray.origin[1] ** 2 - 2 * ray.origin[1] * sphere.center[1] + sphere.center[1] ** 2)
             + (ray.origin[2] ** 2 - 2 * ray.origin[2] * sphere.center[2] + sphere.center[2] ** 2)
             - sphere.radius ** 2)

        # Calculate the discriminant: discriminant = b^2 -4c
        discriminant = b ** 2 - 4 * c

        # If invalid discriminant, return -1
        if discriminant < 0:
            return -1

        t = -1
        # Calculate smaller intersection parameter t0
        t0 = (-b-np.sqrt(discriminant)) / 2

        # If t0 <= 0, then calc the larger t value t1
        if t0 <= 0:
            t1 = (-b + np.sqrt(discriminant)) / 2
            # If t1 <= 0, then it's invalid
            if t1 <= 0:
                return -1
            else:
                t = t1
        else:
            t = t0
        return t

    def triangle_intersection(self, triangle, ray):
        v1v2 = triangle.v2 - triangle.v1
        v1v3 = triangle.v3 - triangle.v1
        N = np.cross(v1v2, v1v3)

        parallelCheck = np.dot(N, ray.direction)
        if abs(parallelCheck) < sys.float_info.epsilon:
            return -1
        D = -np.dot(N, triangle.v1)
        t = -(np.dot(N, ray.origin)+D) / np.dot(N, ray.direction)
        if t < 0:
            return -1

        p = ray.origin + ray.direction * t

        # Check if it's within edge 1
        edge1 = triangle.v2 - triangle.v1
        vp1 = p - triangle.v1
        C = np.cross(edge1, vp1)
        if np.dot(N, C) < 0:
            return -1

        # Check if it's within edge 2
        edge2 = triangle.v3 - triangle.v2
        vp2 = p - triangle.v2
        C = np.cross(edge2, vp2)
        if np.dot(N, C) < 0:
            return -1

        # Check if it's within edge 3
        edge3 = triangle.v1 - triangle.v3
        vp3 = p - triangle.v3
        C = np.cross(edge3, vp3)
        if np.dot(N, C) < 0:
            return -1

        # If all tests pass
        return t

    # Runs through all objects and check if there are any intersections
    def nearest_obj_intersection(self, ray):
        # Gets all distances if there are any intersections
        sphere_dists = [self.sphere_intersection(sphere, ray) for sphere in self.spheres]
        triangle_dists = [self.triangle_intersection(triangle, ray) for triangle in self.triangles]
        dists = sphere_dists + triangle_dists

        # Sets default return vals
        # closest_object is the sphere closest to the camera
        # t is the distance between the camera and the closest_object
        closest_object = None
        t = float('inf')
        isSphere = True

        # If there's actually an intersection
        if not all(dist == -1 for dist in dists):
            # Find the closest point and associated object
            for i in range(len(dists)):
                if 0 < dists[i] < t:
                    t = dists[i]
                    if dists.index(t) < len(self.spheres):
                        closest_object = self.spheres[i]
                    else:
                        closest_object = self.triangles[i-len(self.spheres)]
                        isSphere = False
        # return the closest sphere and the intersection
        return closest_object, t, isSphere

    def reflection_ray(self, V, N):
        return V - 2 * np.dot(V, N) * N

    # Calculates the color of an intersecting object using the Phong Model
    def phong_model(self, object, N, L, V, ray, intersection):
        pixel_color = np.array([0.0, 0.0, 0.0])

        # Check if the point is in shadow. If it is, return black
        shadowRay = Ray()
        shadowRay.origin = intersection
        shadowRay.direction = L
        objectShadowIntersection, _, __ = self.nearest_obj_intersection(shadowRay)
        if objectShadowIntersection is not None and objectShadowIntersection is not object:  # Shadows
            return pixel_color


        ambient = object.ka * self.ambientLight * object.od
        pixel_color += ambient

        # Check for if the dot product is greater than 0
        if np.dot(N, L) > 0:
            diffMax = np.dot(N, L)
        else:
            diffMax = 0
        diffuse = object.kd * self.lightColor * object.od * diffMax
        pixel_color += diffuse

        R = 2 * np.dot(L, N) * N - L
        # Check for if the dot product is greater than 0
        if np.dot(V, R) > 0:
            specMax = np.dot(V, R)
        else:
            specMax = 0
        spec = object.ks * self.lightColor * object.os * specMax ** object.kgls
        pixel_color += spec

        reflection_color = np.array([0.0, 0.0, 0.0])  # Holds the reflection color to be added
        if object.refl != 0.0:  # If the object has reflection

            # First, make the reflection ray
            reflectionRay = Ray()
            reflectionRay.origin = intersection
            reflectionRay.direction = self.reflection_ray(ray.direction, N)

            # Check if the ray intersects another object
            intersectingObject, t, isSphere = self.nearest_obj_intersection(reflectionRay)

            # If nothing is intersected or it intersects with itself,
            # then return the color of the background to be added
            if intersectingObject is None or intersectingObject is object:
                reflection_color += self.backgroundColor
            else:
                # Calculate the color of the new object
                reflection_intersection = reflectionRay.origin + reflectionRay.direction * t
                if isSphere:
                    N = self.normalize(intersection - intersectingObject.center)
                else:
                    v1v2 = intersectingObject.v2 - intersectingObject.v1
                    v1v3 = intersectingObject.v3 - intersectingObject.v1
                    N = np.cross(v1v2, v1v3)
                nudged = reflection_intersection + N * .0001
                reflection_color += self.phong_model(intersectingObject, N, L, V, reflectionRay, nudged)
                if np.count_nonzero(reflection_color) == 0:
                    return reflection_color
        return pixel_color + object.refl*reflection_color

    # Handles the ray tracing and determining the color of the pixel
    def ray_trace(self, ray, i, j):
        # If there's an object(s) blocking the ray, get the closest one
        closest_object, t, isSphere = self.nearest_obj_intersection(ray)

        # If sphere is found, calculate its color
        if closest_object is not None:
            intersection = ray.origin + ray.direction * t
            intersection_to_light = np.linalg.norm(-self.directionToLight)
            if t < intersection_to_light:
                self.image[i][j] = np.clip([0, 0, 0], 0, 1)
            if isSphere:
                N = self.normalize(intersection - closest_object.center)
            else:
                v1v2 = closest_object.v2 - closest_object.v1
                v1v3 = closest_object.v3 - closest_object.v1
                N = np.cross(v1v2, v1v3)
            L = self.normalize(self.directionToLight)
            V = self.normalize(-ray.direction)
            nudged = intersection + N * .0001
            pixel_color = self.phong_model(closest_object, N, L, V, ray, nudged)
        else:  # Otherwise paint the background color
            pixel_color = self.backgroundColor
        self.image[i][j] = np.clip(pixel_color, 0, 1)

    # Handles the intersections and rendering. Writes the output using plt.imsave and saves as a new png file
    def render(self):
        # Sets a default screen (all corners of the screen) to iterate across
        screen = (-1, 1, 1, -1)

        # For x and y
        for i, y in enumerate(np.linspace(screen[1], screen[3], self.image_height)):
            for j, x in enumerate(np.linspace(screen[0], screen[2], self.image_width)):
                self.curr_bounce = 0
                # Define the pixel we'll be working with (Only used for defining the ray's direction)
                pixel = np.array([x, y, 0])

                # Define the ray
                ray = Ray()
                ray.origin = self.cameraLookFrom
                ray.direction = self.normalize(pixel - ray.origin)

                # Ray trace the point with the new ray and the coordinates
                self.ray_trace(ray, i, j)

        plt.imsave('program_5-image.png', self.image)


# Main Function Calls
# Basically, get the file input, read it in, and render it out
file_name = input("Enter the file name to ray trace >> ")
tracer = RayTracer()
tracer.read_image(file_name)
tracer.render()
