#include <vector>
#include "linalg.h"

using namespace std;

struct Mesh
{
public:
    //Empty Constructor
    Mesh()
    {
    }

    //Construct Mesh from OBJ 
    //Mesh(const json &j);

    bool empty() const
    {
        return vs.empty() || f_to_vs.empty();
    }

  vector<linalg::vec<float, 3>>                      vs;      //Vertices
  vector<linalg::vec<float, 2>>                      bs;      //Barycentric Cooridnates
  vector<linalg::vec<int32_t, 3>>                      f_to_vs; //Face to Vertices Map
};
