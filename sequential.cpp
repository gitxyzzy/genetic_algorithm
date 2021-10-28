#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#define SZ(a) ((int)a.size())

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

void mutate_drawing(Drawing& d) {
    int n = SZ(d.polygons);
    int i = random_int(0, n - 1);
    bool modified = true;

    int r = random_int(0, mut_total);
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

double compute_fitness(const cv::Mat& img1, const cv::Mat& img2) {
    double e = 0;
    int n = CANVAS_SZ;
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

    if (random_double() <= mutation_rate) {
        mutate_drawing(off1);
    }

    if (random_double() <= mutation_rate) {
        mutate_drawing(off2);
    }
}

cv::Mat evolve(const cv::Mat& orig_img) {
    Group g = random_group();

    for (int i = 0; i < epochs; i++) {
        for (Drawing& d: g.drawings) {
            if (d.fitness < 0) {
                cv::Mat img = render(d);
                d.fitness = compute_fitness(img, orig_img);
            }
        }

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

    cv::Mat Result = evolve(orig_image);
    cv::imwrite(fout, Result);
}
