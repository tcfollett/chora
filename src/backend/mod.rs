// Define backend trait
// cpu and gpu should implement backend
// associated storages

pub trait Backend: Clone {
    type Storage: Clone;
}
