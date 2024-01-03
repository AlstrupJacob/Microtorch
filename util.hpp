#include <vector>
#include <random>

struct moons {

  std::vector<std::pair<float, float>> pts;
  std::vector<int> clr;
};

moons makemoons() { // points are hardcoded for purpose of example,
//                     may implement more flexible version in the future
  std::vector<std::pair<float, float>> pts;
  std::vector<int> clr;

  moons moons;

  std::vector<std::pair<float, float>> apts{

    std::pair<float, float>{-9.0F, -0.1F},
    std::pair<float, float>{-8.0F, -1.4F},
    std::pair<float, float>{-7.0F, -2.5F},
    std::pair<float, float>{-6.0F, -3.4F},
    std::pair<float, float>{-5.0F, -4.1F},
    std::pair<float, float>{-4.0F, -4.6F},
    std::pair<float, float>{-3.0F, -4.9F},
    std::pair<float, float>{-2.0F, -5.0F},
    std::pair<float, float>{-1.0F, -4.9F},
    std::pair<float, float>{0.0F, -4.6F},
    std::pair<float, float>{1.0F, -4.1F},
    std::pair<float, float>{2.0F, -3.4F},
    std::pair<float, float>{3.0F, -2.5F},
    std::pair<float, float>{4.0F, -1.4F},
    std::pair<float, float>{5.0F, -0.1F},
  };

  std::vector<std::pair<float, float>> bpts{

    std::pair<float, float>{-5.0F, 0.1F},
    std::pair<float, float>{-4.0F, 1.4F},
    std::pair<float, float>{-3.0F, 2.5F},
    std::pair<float, float>{-2.0F, 3.4F},
    std::pair<float, float>{-1.0F, 4.1F},
    std::pair<float, float>{0.0F, 4.6F},
    std::pair<float, float>{1.0F, 4.9F},
    std::pair<float, float>{2.0F, 5.0F},
    std::pair<float, float>{3.0F, 4.9F},
    std::pair<float, float>{4.0F, 4.6F},
    std::pair<float, float>{5.0F, 4.1F},
    std::pair<float, float>{6.0F, 3.4F},
    std::pair<float, float>{7.0F, 2.5F},
    std::pair<float, float>{8.0F, 1.4F},
    std::pair<float, float>{9.0F, 0.1F},
  };

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(-0.5, 0.5);

  for (int i = 0; i < 15; i++) {

    for (int j = 0; j < 10; j++) {

      std::pair<float, float> arv{dis(gen), dis(gen)};
      std::pair<float, float> anp{
                
        arv.first + apts[i].first,
        arv.second + apts[i].second,
      };

      pts.push_back(anp);
      clr.push_back(1);

      std::pair<float, float> brv{dis(gen), dis(gen)};

      std::pair<float, float> bnp{

        brv.first + bpts[i].first,
        brv.second + bpts[i].second,
      };

      pts.push_back(bnp);
      clr.push_back(-1);
    }
  }

  moons.pts = pts;
  moons.clr = clr;

  return moons;
}