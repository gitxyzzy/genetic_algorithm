#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <mpi.h>
#include <opencv2/opencv.hpp>

#define SZ(a) ((int)a.size())

////////////////////////////////////////////////////////////////////////////////
/// MPI
////////////////////////////////////////////////////////////////////////////////

int mpi_rank;
int mpi_size;

////////////////////////////////////////////////////////////////////////////////
/// Globals
////////////////////////////////////////////////////////////////////////////////

// cli
const char* fin = "input.jpg";
const char* fout = "output.jpg";
int epochs = 100;

// image
int img_width = 200;
int img_height = 200;
int img_channels = 4;

#define CANVAS_SZ (img_width * img_height * img_channels)

// genetic algorithm params
int drawings_per_group = 5;

int min_points_per_polygon = 3;
int max_points_per_polygon = 6;

int initial_polygons_per_drawing = 50;
int min_polygons_per_drawing = 2;
int max_polygons_per_drawing = 500;

double mutation_rate = 0.7;
double crossover_rate = 0.3;

////////////////////////////////////////////////////////////////////////////////
/// Random Numbers
////////////////////////////////////////////////////////////////////////////////

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dist_int(0, 255);
std::uniform_real_distribution<double> dist_double(0.0, 1.0);

int random_int(int minVal, int maxVal) {
    return minVal + std::rand() % (maxVal - minVal + 1);
}

double random_double() {
    return ((double)std::rand()) / ((double)RAND_MAX);
}

int random_color_value() {
    return dist_int(gen);
}

double random_alpha() {
    return dist_double(gen);
}

////////////////////////////////////////////////////////////////////////////////
/// Color
////////////////////////////////////////////////////////////////////////////////

struct Color {
    int r, g, b;
    double a;
};

Color random_color() {
    return Color {
        .r = random_color_value(),
        .g = random_color_value(),
        .b = random_color_value(),
        .a = random_alpha(),
    };
}

////////////////////////////////////////////////////////////////////////////////
/// Polygon
////////////////////////////////////////////////////////////////////////////////

struct Polygon {
    std::vector<cv::Point> points;
    Color color;
};

Polygon random_polygon() {
    Polygon p {
        .points {},
        .color = random_color(),
    };

    for (int i = 0; i < min_points_per_polygon; i++) {
        p.points.emplace_back(random_int(0, img_width), random_int(0, img_height));
    }

    return p;
}

void move_polygon(Polygon& p) {
    int max_x = 0, min_x = INT_MAX;
    int max_y = 0, min_y = INT_MAX;

    for (cv::Point& pt : p.points) {
        max_x = std::max(max_x, pt.x);
        min_x = std::min(min_x, pt.x);
        max_y = std::max(max_y, pt.y);
        min_y = std::min(min_y, pt.y);
    }

    int w = max_x - min_x;
    int h = max_y - min_y;

    int new_max_x = w + random_int(0, img_width - w);
    int new_min_y = random_int(0, img_height - h);

    int delta_x = new_max_x - max_x;
    int delta_y = new_min_y - min_y;

    for (cv::Point& pt : p.points) {
        pt.x += delta_x;
        pt.y += delta_y;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Drawing
////////////////////////////////////////////////////////////////////////////////

struct Drawing {
    std::vector<Polygon> polygons;
    double fitness { -1.0 };
};

Drawing random_drawing() {
    Drawing d;
    for (int i = 0; i < initial_polygons_per_drawing; i++) {
        d.polygons.push_back(random_polygon());
    }
    return d;
}

std::vector<int> serialize_drawing(const Drawing& d, int index) {
    int size = 0;
    size += 1; // index
    size += 1; // num polygons
    for (auto& p : d.polygons) {
        size += 1; // num points
        size += SZ(p.points) * 2; // (x, y) of all points
        size += 4; // color
    }

    auto bytes = std::vector<int>(size);
    int i = 0;
    bytes[i++] = index;
    bytes[i++] = SZ(d.polygons);
    for (auto& p : d.polygons) {
        bytes[i++] = SZ(p.points);
        for (auto& pt : p.points) {
            bytes[i++] = pt.x;
            bytes[i++] = pt.y;
        }
        bytes[i++] = p.color.r;
        bytes[i++] = p.color.g;
        bytes[i++] = p.color.b;
        bytes[i++] = int(p.color.a * 1000);
    }

    return bytes;
}

int mut_add_polygon = 1;
int mut_remove_polygon = 1;
int mut_move_polygon = 10;
int mut_change_red = 5;
int mut_change_green = 5;
int mut_change_blue = 5;
int mut_change_alpha = 5;
int mut_add_point = 1;
int mut_remove_point = 1;
int mut_move_x = 5;
int mut_move_y = 5;

int mut_total = mut_add_polygon
    + mut_remove_polygon
    + mut_move_polygon
    + mut_change_red
    + mut_change_green
    + mut_change_blue
    + mut_change_alpha
    + mut_add_point
    + mut_remove_point
    + mut_move_x
    + mut_move_y;

int mut_evade = 2;

void mutate_drawing(Drawing& d) {
    int n = SZ(d.polygons);
    int i = random_int(0, n - 1);
    bool modified = true;

    int r = random_int(0, mut_total + mut_evade);
    if ((r -= mut_add_polygon) <= 0) {
        if (n < max_polygons_per_drawing) {
            d.polygons.insert(d.polygons.begin() + i, random_polygon());
        } else {
            modified = false;
        }
    } else if ((r -= mut_remove_polygon) <= 0) {
        if (n > min_polygons_per_drawing) {
            d.polygons.erase(d.polygons.begin() + i);
        } else {
            modified = false;
        }
    } else {
        Polygon& p = d.polygons[i];
        if ((r -= mut_move_polygon) <= 0) {
            move_polygon(p);
        } else if ((r -= mut_change_red) <= 0) {
            p.color.r = random_int(0, 255);
        } else if ((r -= mut_change_green) <= 0) {
            p.color.g = random_int(0, 255);
        } else if ((r -= mut_change_blue) <= 0) {
            p.color.b = random_int(0, 255);
        } else if ((r -= mut_change_alpha) <= 0) {
            p.color.a = random_int(0, 255);
        } else {
            int npts = SZ(p.points);
            int j = random_int(0, npts - 1);

            if ((r -= mut_add_point) <= 0) {
                if (npts < max_points_per_polygon) {
                    cv::Point pt(random_int(0, img_width), random_int(0, img_height));
                    p.points.insert(p.points.begin() + j, pt);
                } else {
                    modified = false;
                }
            } else if ((r -= mut_remove_point) <= 0) {
                if (npts > min_points_per_polygon) {
                    p.points.erase(p.points.begin() + j);
                } else {
                    modified = false;
                }
            } else {
                cv::Point& pt = p.points[j];
                if ((r -= mut_move_x) <= 0) {
                    pt.x = random_int(0, img_width);
                } else if ((r -= mut_move_y) <= 0) {
                    pt.y = random_int(0, img_height);
                } else {
                    modified = false;
                }
            }
        }
    }

    if (modified) {
        d.fitness = -1.0;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Group
////////////////////////////////////////////////////////////////////////////////

struct Group {
    std::vector<Drawing> drawings;
};

Group random_group() {
    Group g;
    for (int i = 0; i < drawings_per_group; i++) {
        g.drawings.push_back(random_drawing());
    }
    return g;
}

////////////////////////////////////////////////////////////////////////////////
/// Genetic Algorithm
////////////////////////////////////////////////////////////////////////////////

bool fitness_cmp_desc(const Drawing& d1, const Drawing& d2) {
    return d1.fitness > d2.fitness;
}

cv::Mat render(const Drawing& d) {
    cv::Mat image(img_height, img_width, CV_8UC(img_channels), cv::Scalar(0));
    for (const Polygon& p : d.polygons) {
        cv::Mat part = image.clone();
        cv::Scalar color(p.color.b, p.color.g, p.color.r, 255.0);
        int npts = SZ(p.points);
        const cv::Point* pts = p.points.data();
        cv::fillPoly(part, &pts, &npts, 1, color, cv::LINE_4);
        cv::addWeighted(part, p.color.a, image, 1.0 - p.color.a, 0, image);
    }
    return image;
}

cv::Mat render_serialized_drawing(const int* bytes) {
    cv::Mat image(img_height, img_width, CV_8UC(img_channels), cv::Scalar(0));

    int i = 0;
    int index = bytes[i++];
    int num_polygons = bytes[i++];

    for (int j = 0; j < num_polygons; j++) {
        cv::Mat part = image.clone();

        int num_points = bytes[i++];
        std::vector<cv::Point> points;
        points.reserve(num_points);
        for (int k = 0; k < num_points; k++) {
            int x = bytes[i++];
            int y = bytes[i++];
            points.emplace_back(x, y);
        }

        int r = bytes[i++];
        int g = bytes[i++];
        int b = bytes[i++];
        double a = bytes[i++] / 1000.0;

        cv::Scalar color(b, g, r, 255.0);
        const cv::Point* pts = points.data();
        cv::fillPoly(part, &pts, &num_points, 1, color, cv::LINE_4);
        cv::addWeighted(part, a, image, 1.0 - a, 0, image);
    }

    return image;
}

double compute_fitness(const cv::Mat& img1, const cv::Mat& img2) {
    double e = 0;
    const int n = CANVAS_SZ;
    for (int i = 0; i < n; i++) {
        e += std::abs(img1.data[i] - img2.data[i]);
    }
    return 1 - (e / (255.0 * n));
}

int fitness_proportionate_selection(int n) {
    double sum = n * (n + 1) / 2.0;
    double ptr = 1.0;
    double stop = random_double();
    int last = n - 1;
    for (int i = 0; i < last; i++, n--) {
        ptr -= n / sum;
        if (ptr <= stop) {
            return i;
        }
    }
    return last;
}

void two_point_crossover(const Drawing& par1, const Drawing& par2, Drawing& off1, Drawing& off2) {
    const Drawing& fittest = par1.fitness > par2.fitness ? par1 : par2;
    int n = std::min(SZ(par1.polygons), SZ(par2.polygons));
    int r1 = random_int(0, n - 1);
    int r2 = random_int(0, n - 1);
    int i1 = std::min(r1, r2);
    int i2 = std::max(r1, r2);

    for (int i = 0; i < i1; i++) {
        off1.polygons.push_back(fittest.polygons[i]);
        off2.polygons.push_back(fittest.polygons[i]);
    }

    for (int i = i1; i <= i2; i++) {
        off1.polygons.push_back(par2.polygons[i]);
        off2.polygons.push_back(par1.polygons[i]);
    }

    for (int i = i2 + 1; i < SZ(fittest.polygons); i++) {
        off1.polygons.push_back(fittest.polygons[i]);
        off2.polygons.push_back(fittest.polygons[i]);
    }
}

void mate(const Drawing& par1, const Drawing& par2, Drawing& off1, Drawing& off2) {
    double r = random_double();
    if (r <= crossover_rate) {
        two_point_crossover(par1, par2, off1, off2);
    } else {
        off1 = par1;
        off2 = par2;
    }

    mutate_drawing(off1);
    mutate_drawing(off2);
}

////////////////////////////////////////////////////////////////////////////////
/// Workers
////////////////////////////////////////////////////////////////////////////////

enum WorkType : int {
    k_compute_drawing_fitness = 0,
    k_exit,
};

void workers_entry(const cv::Mat& orig_img) {
    while (true) {
        int work_type;
        MPI_Recv(&work_type, 1, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (work_type == k_compute_drawing_fitness) {
            int buf_size;
            MPI_Recv(&buf_size, 1, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            auto buffer = std::make_unique<int[]>(buf_size);
            MPI_Recv(buffer.get(), buf_size, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int index = buffer[0];
            auto img = render_serialized_drawing(buffer.get());
            double fitness = compute_fitness(img, orig_img);

            MPI_Send(&index, 1, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD);
            MPI_Send(&fitness, 1, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD);
        } else if (work_type == k_exit) {
            break;
        }
    }
}

void workers_stop() {
    int work_type = k_exit;
    for (int i = 0; i < mpi_size; i++) {
        MPI_Send(&work_type, 1, MPI_INT, i, i, MPI_COMM_WORLD);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Genetic Algorithm
////////////////////////////////////////////////////////////////////////////////

void init_fitness(Group& g) {
    std::vector<bool> available(mpi_size, true);
    std::vector<MPI_Request> requests(mpi_size);
    std::vector<std::vector<int>> serialized_drawings(mpi_size);
    std::vector<int> receive_buffers(mpi_size);
    int work_type = k_compute_drawing_fitness;

    int drawings_processed = 0;
    int total_drawings = SZ(g.drawings);
    int next_index = 0;

    while (drawings_processed < total_drawings) {
        for (int i = 1; i < mpi_size; i++) {
            if (available[i]) {
                while (next_index < total_drawings && g.drawings[next_index].fitness >= 0) {
                    drawings_processed++;
                    next_index++;
                }

                if (next_index < total_drawings) {
                    available[i] = false;
                    serialized_drawings[i] = serialize_drawing(g.drawings[next_index], next_index);
                    int buf_size = SZ(serialized_drawings[i]);
                    MPI_Send(&work_type, 1, MPI_INT, i, i, MPI_COMM_WORLD);
                    MPI_Send(&buf_size, 1, MPI_INT, i, i, MPI_COMM_WORLD);
                    MPI_Send(serialized_drawings[i].data(), buf_size, MPI_INT, i, i, MPI_COMM_WORLD);
                    MPI_Irecv(&receive_buffers[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, &requests[i]);
                    next_index++;
                }
            } else {
                int did_complete;
                MPI_Test(&requests[i], &did_complete, MPI_STATUS_IGNORE);
                if (did_complete) {
                    int index = receive_buffers[i];
                    MPI_Recv(&g.drawings[index].fitness, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    drawings_processed++;
                    serialized_drawings[i].clear();
                    available[i] = true;
                }
            }
        }
    }
}

cv::Mat evolve(const cv::Mat& orig_img) {
    Group g = random_group();

    for (int i = 0; i < epochs; i++) {
        init_fitness(g);

        std::sort(g.drawings.begin(), g.drawings.end(), fitness_cmp_desc);

        std::vector<Drawing> next { g.drawings.front() };
        next.resize(1 + (drawings_per_group / 2) * 2);

        for (int k = 1; k < SZ(next); k += 2) {
            int p1 = fitness_proportionate_selection(SZ(g.drawings));
            int p2 = fitness_proportionate_selection(SZ(g.drawings));
            mate(g.drawings[p1], g.drawings[p2], next[k], next[k + 1]);
        }

        g.drawings = std::move(next);
    }

    std::sort(g.drawings.begin(), g.drawings.end(), fitness_cmp_desc);
    return render(g.drawings.front());
}

////////////////////////////////////////////////////////////////////////////////
/// main()
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    std::srand(std::time(nullptr));

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-i") == 0) {
            fin = argv[++i];
        } else if (strcmp(argv[i], "-e") == 0) {
            epochs = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0) {
            fout = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0) {
            std::srand(std::stoi(argv[++i]));
        }
    }

    cv::Mat orig_image = cv::imread(fin, cv::IMREAD_UNCHANGED);
    if (!orig_image.data) {
        std::cout << "Image " << fin << " not found.\n";
        return 0;
    }

    cv::Size size = orig_image.size();
    img_width = size.width;
    img_height = size.height;
    img_channels = orig_image.channels();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_rank == 0) {
        cv::Mat Result = evolve(orig_image);
        cv::imwrite(fout, Result);
        workers_stop();
    } else {
        workers_entry(orig_image);
    }

    MPI_Finalize();
}
