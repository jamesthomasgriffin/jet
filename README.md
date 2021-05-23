# jet

Templated implementation of jets from differential calculus.

Suppose we have some parameters, e.g. `x`, `y` and `scale` and that these are used to compute a position `p`.  A jet is exactly the structure required to replace `p` with not only the calculated position, but also the information about how `p` changes as the parameters are changed.

Typically any quantity represented by floating point numbers can be replaced by a jet, and any function that transforms such types can also transform jets.  This library exists to make this as simple as possible.

The main template is Jet2, this defines classes:

Jet2<V, P>

which have values of type V and scalar parameters labelled by type P, a common choice for P might be float* or double* (the type only needs to implement comparison, it is never dereferenced).  This jet holds both first and second derivative information.
